#!/usr/bin/env python3
"""AI-powered issue triage for the polkit project using Gemini."""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
import re
import requests

from polkit_context import (
    POLKIT_SUMMARY,
    PROMPT_ASSESS,
    PROMPT_DESIGN_REPRODUCER,
    PROMPT_DESIGN_SOLUTION,
    PROMPT_ELICIT,
    PROMPT_LABEL,
    PROMPT_VALIDATE_DOCKERFILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ai_issue_triage")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DOCKER_TIMEOUT_SECONDS = 300
DOCKER_MEMORY_LIMIT = "512m"
MAX_COMMENT_LENGTH = 65536

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

_TRIAGE_MARKER = "Issue triaged by AI assistant"
_TRIAGE_MARKER_BOT = "github-actions[bot]"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

def _stripc_fences(text: str) -> str:
    m = _FENCE_RE.search(text.strip())
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResult:
    type: str = "unknown"
    confidence: float = 0.0
    summary: str = ""
    missing_info: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)


@dataclass
class ReproducerDesign:
    reproducer_script: str = ""
    script_filename: str = "reproducer.sh"
    base_image: str = "fedora:latest"
    extra_packages: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class SolutionDesign:
    approach: str = ""
    affected_files: list[str] = field(default_factory=list)
    complexity: str = "unknown"
    security_considerations: list[str] = field(default_factory=list)
    sketch: str = ""


@dataclass
class DesignResult:
    kind: str = ""  # "reproducer" or "solution"
    reproducer: ReproducerDesign | None = None
    solution: SolutionDesign | None = None


@dataclass
class ValidationResult:
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    dockerfile: str = ""


# ---------------------------------------------------------------------------
# Gemini REST API client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Thin wrapper around the Gemini REST API with retry logic."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        self.api_key = api_key
        self.model = model
        self._session = requests.Session()

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent"
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if system_instruction:
            body["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }
        body["generationConfig"] = {
            "temperature": 0.2,
            "maxOutputTokens": 16384,
        }

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._session.post(
                    url,
                    headers={"x-goog-api-key": self.api_key},
                    json=body,
                    timeout=120,
                )
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    log.warning("Gemini rate-limited (429), retrying in %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                log.debug("Gemini response (%d chars):\n%s", len(text), text)
                return text
            except Exception as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Gemini API failed after 3 attempts: {last_err}"
        ) from last_err


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------

class GitHubClient:
    """Thin wrapper around the GitHub REST API for issue operations."""

    def __init__(self, token: str, repo: str):
        self.repo = repo
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self._base = f"https://api.github.com/repos/{repo}"

    def get_issue(self, number: int) -> dict:
        resp = self._session.get(f"{self._base}/issues/{number}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_issue_comments(self, number: int) -> list[dict]:
        resp = self._session.get(f"{self._base}/issues/{number}/comments", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def add_labels(self, number: int, labels: list[str]) -> None:
        if not labels:
            return
        resp = self._session.post(
            f"{self._base}/issues/{number}/labels",
            json={"labels": labels},
            timeout=30,
        )
        resp.raise_for_status()
        log.info("Applied labels %s to issue #%d", labels, number)

    def post_comment(self, number: int, body: str) -> None:
        body = body[:MAX_COMMENT_LENGTH]
        resp = self._session.post(
            f"{self._base}/issues/{number}/comments",
            json={"body": body},
            timeout=30,
        )
        resp.raise_for_status()
        log.info("Posted comment on issue #%d", number)

    def get_labels(self) -> list[str]:
        labels: list[str] = []
        page = 1
        while True:
            resp = self._session.get(
                f"{self._base}/labels",
                params={"per_page": 100, "page": page},
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            labels.extend(item["name"] for item in batch)
            page += 1
        return labels


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _fix_json_escapes(text: str) -> str:
    """Fix invalid JSON escape sequences without breaking already-valid ones.

    Walks the string character-by-character so that already-escaped
    backslashes (``\\\\``) are consumed as pairs and left intact, while
    truly invalid escapes like ``\\$`` or ``\\x`` get an extra backslash.
    """
    _VALID_AFTER_BS = set('"\\/bfnrtu')
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in _VALID_AFTER_BS:
                out.append(text[i:i + 2])
                i += 2
                if nxt == 'u' and i + 4 <= len(text):
                    out.append(text[i:i + 4])
                    i += 4
            else:
                # Invalid escape — double the backslash, leave next char
                out.append('\\\\')
                i += 1
        else:
            out.append(ch)
            i += 1
    return ''.join(out)

def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from Gemini's response, tolerating markdown fences."""
    text = _stripc_fences(text)
    # Fallback: if text doesn't look like JSON, extract the first JSON object
    if text and not text.startswith(('{', '[')):
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end + 1]
    # Gemini sometimes emits invalid JSON escapes (e.g. \$, \x03) inside strings.
    text = _fix_json_escapes(text)
    return json.loads(text)


def _safe_parse_json(text: str, context: str) -> dict | None:
    try:
        return _parse_json_response(text)
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        log.error("Failed to parse Gemini JSON for %s: %s\nRaw: %s", context, exc, text[:500])
        return None


# ---------------------------------------------------------------------------
# Feature 1: Assess
# ---------------------------------------------------------------------------

def assess(gemini: GeminiClient, issue: dict) -> AssessmentResult | None:
    log.info("Assessing issue #%s: %s", issue["number"], issue["title"])
    prompt = PROMPT_ASSESS.format(
        issue_title=issue["title"],
        issue_body=issue.get("body", "") or "",
    )
    raw = gemini.generate(prompt, system_instruction=POLKIT_SUMMARY)
    data = _safe_parse_json(raw, "assess")
    if data is None:
        return None

    valid_types = {"bug", "feature_request", "ci_cd", "question", "invalid"}
    issue_type = data.get("type", "unknown")
    if issue_type not in valid_types:
        log.warning("Gemini returned unknown type '%s', defaulting to 'unknown'", issue_type)
        issue_type = "unknown"

    return AssessmentResult(
        type=issue_type,
        confidence=float(data.get("confidence", 0.0)),
        summary=data.get("summary", ""),
        missing_info=data.get("missing_info", []),
        affected_components=data.get("affected_components", []),
    )


# ---------------------------------------------------------------------------
# Feature 2: Label
# ---------------------------------------------------------------------------

def label(
    gemini: GeminiClient,
    github: GitHubClient,
    issue: dict,
    assessment: AssessmentResult,
) -> list[str]:
    log.info("Labeling issue #%s", issue["number"])
    available = github.get_labels()
    if not available:
        log.warning("No labels found in repo, skipping labeling")
        return []

    prompt = PROMPT_LABEL.format(
        available_labels=json.dumps(available),
        assessment_json=json.dumps({
            "type": assessment.type,
            "confidence": assessment.confidence,
            "summary": assessment.summary,
            "affected_components": assessment.affected_components,
        }, indent=2),
    )
    raw = gemini.generate(prompt, system_instruction=POLKIT_SUMMARY)
    data = _safe_parse_json(raw, "label")
    if data is None:
        return []

    suggested = data.get("labels", [])
    available_set = set(available)
    validated = [lbl for lbl in suggested if lbl in available_set]
    rejected = [lbl for lbl in suggested if lbl not in available_set]
    if rejected:
        log.warning("Rejected non-existent labels: %s", rejected)

    github.add_labels(issue["number"], validated)
    return validated


# ---------------------------------------------------------------------------
# Feature 3: Elicit
# ---------------------------------------------------------------------------

def elicit(
    gemini: GeminiClient,
    github: GitHubClient,
    issue: dict,
    assessment: AssessmentResult,
) -> str | None:
    if not assessment.missing_info:
        log.info("No missing info for issue #%s, skipping elicitation", issue["number"])
        return None

    log.info("Requesting missing info for issue #%s: %s", issue["number"], assessment.missing_info)
    prompt = PROMPT_ELICIT.format(
        issue_title=issue["title"],
        issue_body=issue.get("body", "") or "",
        missing_info="\n".join(f"- {item}" for item in assessment.missing_info),
    )
    comment_text = gemini.generate(prompt, system_instruction=POLKIT_SUMMARY)
    github.post_comment(issue["number"], comment_text)
    return comment_text


# ---------------------------------------------------------------------------
# Feature 4: Design
# ---------------------------------------------------------------------------

def design_reproducer(
    gemini: GeminiClient,
    issue: dict,
    assessment: AssessmentResult,
) -> ReproducerDesign | None:
    log.info("Designing reproducer for issue #%s", issue["number"])
    prompt = PROMPT_DESIGN_REPRODUCER.format(
        issue_title=issue["title"],
        issue_body=issue.get("body", "") or "",
        assessment_json=json.dumps({
            "type": assessment.type,
            "summary": assessment.summary,
            "affected_components": assessment.affected_components,
        }, indent=2),
    )
    raw = gemini.generate(prompt, system_instruction=POLKIT_SUMMARY)
    data = _safe_parse_json(raw, "design_reproducer")
    if data is None:
        return None

    return ReproducerDesign(
        reproducer_script=data.get("reproducer_script", ""),
        script_filename=data.get("script_filename", "reproducer.sh"),
        base_image=data.get("base_image", "fedora:latest"),
        extra_packages=data.get("extra_packages", []),
        explanation=data.get("explanation", ""),
    )


def design_solution(
    gemini: GeminiClient,
    issue: dict,
    assessment: AssessmentResult,
) -> SolutionDesign | None:
    log.info("Designing solution for issue #%s", issue["number"])
    prompt = PROMPT_DESIGN_SOLUTION.format(
        issue_title=issue["title"],
        issue_body=issue.get("body", "") or "",
        assessment_json=json.dumps({
            "type": assessment.type,
            "summary": assessment.summary,
            "affected_components": assessment.affected_components,
        }, indent=2),
    )
    raw = gemini.generate(prompt, system_instruction=POLKIT_SUMMARY)
    data = _safe_parse_json(raw, "design_solution")
    if data is None:
        return None

    return SolutionDesign(
        approach=data.get("approach", ""),
        affected_files=data.get("affected_files", []),
        complexity=data.get("complexity", "unknown"),
        security_considerations=data.get("security_considerations", []),
        sketch=data.get("sketch", ""),
    )


def design(
    gemini: GeminiClient,
    issue: dict,
    assessment: AssessmentResult,
) -> DesignResult | None:
    if assessment.type == "bug":
        repro = design_reproducer(gemini, issue, assessment)
        if repro:
            return DesignResult(kind="reproducer", reproducer=repro)
    elif assessment.type == "feature_request":
        sol = design_solution(gemini, issue, assessment)
        if sol:
            return DesignResult(kind="solution", solution=sol)
    else:
        log.info(
            "Issue type '%s' is not a bug or feature request, skipping design",
            assessment.type,
        )
    return None


# ---------------------------------------------------------------------------
# Feature 5: Validate
# ---------------------------------------------------------------------------

def validate(
    gemini: GeminiClient,
    github: GitHubClient,
    issue: dict,
    design_result: DesignResult,
) -> ValidationResult | None:
    if design_result.kind != "reproducer" or design_result.reproducer is None:
        log.info("No reproducer to validate for issue #%s", issue["number"])
        return None

    repro = design_result.reproducer
    log.info("Validating reproducer for issue #%s in Docker", issue["number"])

    dockerfile_prompt = PROMPT_VALIDATE_DOCKERFILE.format(
        base_image=repro.base_image,
        extra_packages=" ".join(repro.extra_packages) if repro.extra_packages else "",
        script_filename=repro.script_filename,
    )
    dockerfile_content = gemini.generate(
        dockerfile_prompt, system_instruction=POLKIT_SUMMARY
    )
    dockerfile_content = _stripc_fences(dockerfile_content)

    result = ValidationResult(dockerfile=dockerfile_content)

    with tempfile.TemporaryDirectory(prefix="polkit-validate-") as tmpdir:
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        reproducer_path = os.path.join(tmpdir, repro.script_filename)

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        with open(reproducer_path, "w") as f:
            f.write(repro.reproducer_script)
        os.chmod(reproducer_path, 0o755)

        tag = f"polkit-validate-{issue['number']}"

        try:
            build_proc = subprocess.run(
                [
                    "docker", "build",
                    "-t", tag,
                    "-f", dockerfile_path,
                    tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=DOCKER_TIMEOUT_SECONDS,
            )
            if build_proc.returncode != 0:
                log.error("Docker build failed:\n%s", build_proc.stderr[-2000:])
                result.stderr = build_proc.stderr[-2000:]
                result.exit_code = build_proc.returncode
                return result

            run_proc = subprocess.run(
                [
                    "docker", "run",
                    "--rm",
                    "--network=none",
                    f"--memory={DOCKER_MEMORY_LIMIT}",
                    tag,
                ],
                capture_output=True,
                text=True,
                timeout=DOCKER_TIMEOUT_SECONDS,
            )
            result.exit_code = run_proc.returncode
            result.stdout = run_proc.stdout[-4000:]
            result.stderr = run_proc.stderr[-4000:]
            result.success = run_proc.returncode == 0

        except subprocess.TimeoutExpired:
            log.error("Docker operation timed out after %ds", DOCKER_TIMEOUT_SECONDS)
            result.stderr = f"Timed out after {DOCKER_TIMEOUT_SECONDS}s"
        except FileNotFoundError:
            log.error("Docker not found -- is it installed on this runner?")
            result.stderr = "Docker not found on runner"
        finally:
            subprocess.run(
                ["docker", "rmi", "-f", tag],
                capture_output=True,
                timeout=30,
            )

    return result


# ---------------------------------------------------------------------------
# Feature 6: Communicate
# ---------------------------------------------------------------------------

def _issue_already_has_reproducer(github: GitHubClient, issue: dict) -> bool:
    """Heuristic: check if the issue body contains code blocks that look like a reproducer."""
    body = (issue.get("body") or "").lower()
    comments = github.get_issue_comments(issue["number"])
    sources = body + " ".join((comment.get("body") or "").lower() for comment in comments)
    code_indicators = ["```", "#!/bin/", "reproducer", "steps to reproduce"]
    script_indicators = ["pkexec", "pkcheck", "busctl", "gdbus", "dbus-send"]
    has_code = any(ind in sources for ind in code_indicators)
    has_polkit_tool = any(ind in sources for ind in script_indicators)
    return has_code and has_polkit_tool


def communicate(
    github: GitHubClient,
    issue: dict,
    design_result: DesignResult,
    validation_result: ValidationResult | None = None,
) -> str | None:
    if design_result.kind != "reproducer" or design_result.reproducer is None:
        return None

    if _issue_already_has_reproducer(github, issue):
        log.info("Issue #%s already contains a reproducer, skipping", issue["number"])
        return None

    repro = design_result.reproducer

    # Validation failed or was not run — post a short notice instead
    if validation_result is None or not validation_result.success:
        log.info("Reproducer for issue #%s did not pass validation, posting failure notice", issue["number"])
        parts = ["### Automated Reproducer\n"]
        parts.append(
            "An automated reproducer was generated but **could not be verified** "
            "in an isolated Docker container.\n"
        )
        output = "\n".join(filter(None, [
            validation_result.stdout.strip() if validation_result else "",
            validation_result.stderr.strip() if validation_result else "",
        ]))
        if validation_result and output:
            parts.append(
                f"<details><summary>Validation error details</summary>\n\n"
                f"**Exit code:** {validation_result.exit_code}\n\n"
                f"```\n{output}\n```\n"
                f"</details>\n"
            )
        parts.append(
            "\n---\n"
            "*The AI assistant was unable to produce a working reproducer "
            "within the current constraints. "
            "A maintainer may attempt to reproduce this manually.*"
        )
        comment = "\n".join(parts)
        github.post_comment(issue["number"], comment)
        return comment

    # Validation succeeded — post the verified reproducer
    log.info("Posting verified reproducer for issue #%s", issue["number"])
    extra_pkgs = ""
    if repro.extra_packages:
        extra_pkgs = ", additional packages: " + ", ".join(f"`{p}`" for p in repro.extra_packages)

    comment = (
        "### Verified Reproducer\n\n"
        "The following reproducer was automatically generated and "
        "**successfully validated** in an isolated Docker container.\n\n"
        f"**What it does:** {repro.explanation}\n\n"
        f"**Reproducer** (`{repro.script_filename}`):\n"
        f"```bash\n{repro.reproducer_script}\n```\n\n"
        f"**Environment:** `{repro.base_image}`{extra_pkgs}\n\n"
    )
    if validation_result.stdout.strip():
        comment += (
            "<details><summary>Validation output</summary>\n\n"
            f"```\n{validation_result.stdout.strip()}\n```\n"
            "</details>\n\n"
        )
    if validation_result.dockerfile.strip():
        comment += (
            "<details><summary>Run it locally with Docker/Podman</summary>\n\n"
            f"Save the reproducer script as `{repro.script_filename}` and "
            "the following as `Dockerfile` in the same directory, then run:\n\n"
            "```bash\n"
            f"docker build -t polkit-repro . && docker run --rm polkit-repro\n"
            "```\n\n"
            "Or with Podman:\n\n"
            "```bash\n"
            f"podman build -t polkit-repro . && podman run --rm polkit-repro\n"
            "```\n\n"
            f"**Dockerfile:**\n"
            f"```dockerfile\n{validation_result.dockerfile.strip()}\n```\n"
            "</details>\n\n"
        )
    comment += (
        "---\n"
        "*This reproducer was generated and verified by an AI assistant. "
        "Please confirm it matches the problem you reported.*"
    )

    github.post_comment(issue["number"], comment)
    return comment


# ---------------------------------------------------------------------------
# CLI and pipeline
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-powered issue triage for the polkit project",
    )
    parser.add_argument("--issue-number", type=int, required=True, help="GitHub issue number")
    parser.add_argument("--repo", required=True, help="owner/repo (e.g. polkit-org/polkit)")
    parser.add_argument(
        "--model", default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)",
    )

    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug logging (shows raw Gemini responses)",
    )

    for feat in ("assess", "label", "elicit", "design", "communicate", "validate"):
        parser.add_argument(
            f"--{feat}", action=argparse.BooleanOptionalAction, default=True,
            help=f"Enable/disable the {feat} stage",
        )

    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    gemini = GeminiClient(api_key=GEMINI_API_KEY, model=args.model)
    github = GitHubClient(token=GITHUB_TOKEN, repo=args.repo)
    ret_val = 0

    issue = github.get_issue(args.issue_number)
    log.info("Fetched issue #%d: %s", args.issue_number, issue["title"])

    assessment: AssessmentResult | None = None
    applied_labels: list[str] = []
    design_result: DesignResult | None = None

    # Stage 0: Check if issue is already triaged
    comments = github.get_issue_comments(args.issue_number)
    
    if any(
        comment["user"]["login"] == _TRIAGE_MARKER_BOT
        and _TRIAGE_MARKER in comment["body"]
        for comment in comments
    ):
        log.info("Issue #%d already triaged", args.issue_number)
        ret_val = 1
        return ret_val

    # Stage 1: Assess
    if args.assess:
        try:
            assessment = assess(gemini, issue)
            if assessment:
                log.info(
                    "Assessment: type=%s confidence=%.2f summary=%s",
                    assessment.type, assessment.confidence, assessment.summary,
                )
            else:
                log.warning("Assessment returned no result")
        except Exception:
            log.exception("Assessment failed")
            ret_val = 2
    # Stage 2: Label
    if args.label and assessment:
        try:
            applied_labels = label(gemini, github, issue, assessment)
            log.info("Applied labels: %s", applied_labels)
        except Exception:
            log.exception("Labeling failed")
            ret_val = 2
    else:
        log.info("Skipping labeling: no assessment result")

    # Stage 3: Elicit
    if args.elicit and assessment:
        try:
            elicit(gemini, github, issue, assessment)
        except Exception:
            log.exception("Elicitation failed")
            ret_val = 2
    else:
        log.info("Skipping elicitation: no assessment result")

    # Stage 4: Design
    if args.design and assessment:
        try:
            design_result = design(gemini, issue, assessment)
            if design_result:
                log.info("Design complete: kind=%s", design_result.kind)
        except Exception:
            log.exception("Design failed")
            ret_val = 2
    else:
        log.info("Skipping design: no assessment result")

    # Stage 5: Validate (before communicate — only post verified reproducers)
    validation_result: ValidationResult | None = None
    if args.validate and design_result:
        try:
            validation_result = validate(gemini, github, issue, design_result)
            if validation_result:
                log.info("Validation: success=%s exit_code=%d", validation_result.success, validation_result.exit_code)
        except Exception:
            log.exception("Validation failed")
            ret_val = 2
    else:
        log.info("Skipping validation: no design result")

    # Stage 6: Communicate (posts based on validation outcome)
    if args.communicate and design_result:
        try:
            communicate(github, issue, design_result, validation_result)
        except Exception:
            log.exception("Communication failed")
            ret_val = 2
    else:
        log.info("Skipping communication: no design result")   

    log.info("Pipeline complete for issue #%d", args.issue_number)

    if ret_val == 0 and assessment is not None:
        github.post_comment(args.issue_number, _TRIAGE_MARKER)

    return ret_val


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ret_val = run_pipeline(args)
    sys.exit(ret_val)

if __name__ == "__main__":
    main()
