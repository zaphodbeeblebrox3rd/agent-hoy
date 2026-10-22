# Unattended training sessions.
#
# Rollback: set TRAINING_ENABLED to False. `python main.py --train` then exits
# without launching a session.

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TRAINING_ENABLED = True
TRAINING_DIR = "training"
DEFAULT_TIMEOUT_SECONDS = 180.0
STABLE_SECONDS = 2.5

PROMPTS = (
    "How do I keep older analysis on screen while still seeing the newest line?",
    "Why would a Kubernetes pod stay in Pending?",
    "How do I check whether a Linux disk is full?",
    "What does a non-zero exit code mean in a shell script?",
    "How can I tell if a TLS certificate is expired?",
    "What is the difference between a process and a thread?",
    "How do I find which program is listening on port 443?",
    "Why would an Ansible playbook hang while waiting on a host?",
)


@dataclass(frozen=True)
class TrainArgs:
    train: bool = False
    iterations: int = 1
    training_run: bool = False
    prompt: str = ""
    output: str = ""
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    seed: int | None = None


@dataclass(frozen=True)
class SessionEvaluation:
    prompt_in_transcript: bool
    analysis_present: bool
    analysis_error: bool
    vision_captured: bool
    timed_out: bool

    @property
    def passed(self) -> bool:
        return (
            self.prompt_in_transcript
            and self.analysis_present
            and self.vision_captured
            and not self.analysis_error
            and not self.timed_out
        )


def parse_train_args(argv: list[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(description="Real-time speech transcription")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training iterations, then exit",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="Number of training iterations (default 1)",
    )
    parser.add_argument("--training-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prompt", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for one iteration's analysis",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for prompt choice")
    parsed = parser.parse_args(argv)
    if parsed.iterations < 1:
        parser.error("-n must be at least 1")
    if parsed.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    return TrainArgs(
        train=parsed.train,
        iterations=parsed.iterations,
        training_run=parsed.training_run,
        prompt=parsed.prompt,
        output=parsed.output,
        timeout=parsed.timeout,
        seed=parsed.seed,
    )


def choose_prompts(count: int, seed: int | None = None) -> list[str]:
    picker = random.Random(seed)
    if count <= len(PROMPTS):
        return picker.sample(list(PROMPTS), count)
    return [picker.choice(PROMPTS) for _ in range(count)]


def evaluate_session(
    prompt: str,
    transcript: str,
    ai_text: str,
    vision_error: str | None,
    image_bytes: bytes | None,
    timed_out: bool,
) -> SessionEvaluation:
    analysis = (ai_text or "").strip()
    error_prefixes = (
        "OpenAI API error",
        "Guide turn failed",
        "OpenAI is not configured",
        "AI Analysis disabled",
        "Contextual AI analysis generation failed",
    )
    return SessionEvaluation(
        prompt_in_transcript=bool(prompt) and prompt.lower() in (transcript or "").lower(),
        analysis_present=bool(analysis),
        analysis_error=analysis.startswith(error_prefixes),
        vision_captured=bool(image_bytes) and not vision_error,
        timed_out=timed_out,
    )


def render_session_report(
    *,
    iteration: int,
    prompt: str,
    transcript: str,
    ai_text: str,
    topic_text: str,
    vision_label: str,
    vision_error: str | None,
    image_name: str,
    evaluation: SessionEvaluation,
) -> str:
    if vision_error:
        vision_line = vision_error
    elif image_name and vision_label:
        vision_line = f"{vision_label} saved as {image_name}"
    else:
        vision_line = "No window image was captured."
    lines = [
        f"# Training iteration {iteration}",
        "",
        f"Started: {datetime.now().isoformat(timespec='seconds')}",
        f"Prompt: {prompt}",
        "",
        "## Transcript",
        "",
        transcript.strip() or "(empty)",
        "",
        "## AI pane",
        "",
        ai_text.strip() or "(empty)",
        "",
        "## Topic pane",
        "",
        topic_text.strip() or "(empty)",
        "",
        "## Vision",
        "",
        vision_line,
        "",
        "## Evaluation",
        "",
        f"prompt_in_transcript: {_yes_no(evaluation.prompt_in_transcript)}",
        f"analysis_present: {_yes_no(evaluation.analysis_present)}",
        f"analysis_error: {_yes_no(evaluation.analysis_error)}",
        f"vision_captured: {_yes_no(evaluation.vision_captured)}",
        f"timed_out: {_yes_no(evaluation.timed_out)}",
        f"passed: {_yes_no(evaluation.passed)}",
        "",
        "Read this report and the screenshot, then change the app before the next iteration.",
        "",
    ]
    return "\n".join(lines)


def run_training(args: TrainArgs, main_path: str | None = None) -> int:
    """Launch one child process per iteration. Each child exits on its own."""
    if not TRAINING_ENABLED:
        print("Training is off. Set TRAINING_ENABLED to True in train_mode.py.")
        return 2

    output_dir = Path(TRAINING_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = choose_prompts(args.iterations, args.seed)
    script = main_path or str(Path(__file__).with_name("main.py"))
    reports = []
    failures = 0

    for index, prompt in enumerate(prompts, start=1):
        report_path = output_dir / f"run-{index:03d}.md"
        image_path = output_dir / f"run-{index:03d}.png"
        print(f"Training iteration {index}/{args.iterations}: {prompt}")
        command = [
            sys.executable,
            script,
            "--training-run",
            "--prompt",
            prompt,
            "--output",
            str(report_path),
            "--timeout",
            str(args.timeout),
        ]
        try:
            completed = subprocess.run(
                command,
                timeout=args.timeout + 90,
                check=False,
            )
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
            report_path.write_text(
                render_session_report(
                    iteration=index,
                    prompt=prompt,
                    transcript="",
                    ai_text="",
                    topic_text="",
                    vision_label="",
                    vision_error="The training process exceeded its timeout before writing a report.",
                    image_name=image_path.name,
                    evaluation=evaluate_session(prompt, "", "", "timed out", None, True),
                ),
                encoding="utf-8",
            )
        if code != 0:
            failures += 1
        if report_path.exists():
            reports.append(report_path.read_text(encoding="utf-8"))
        else:
            reports.append(f"# Training iteration {index}\n\nThe child exited {code} without a report.\n")
        latest = output_dir / "latest.md"
        latest.write_text("\n\n".join(reports), encoding="utf-8")
        print(f"Iteration {index} exited {code}. Report: {report_path}")
        print(f"Updated {latest}")

    return 1 if failures else 0


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
