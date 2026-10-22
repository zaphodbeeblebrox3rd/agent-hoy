#!/usr/bin/env python3
"""Tests for the unattended training command."""

import sys

from train_mode import (
    TRAINING_ENABLED,
    choose_prompts,
    evaluate_session,
    parse_train_args,
    render_session_report,
)


def test_parse_train_command():
    args = parse_train_args(["--train", "-n", "3", "--seed", "7"])
    assert args.train is True
    assert args.iterations == 3
    assert args.seed == 7
    assert args.training_run is False
    plain = parse_train_args([])
    assert plain.train is False
    print("OK train arguments")
    return True


def test_prompts_are_distinct_for_a_seed():
    first = choose_prompts(3, seed=7)
    second = choose_prompts(3, seed=7)
    assert first == second
    assert len(set(first)) == 3
    print("OK prompt selection")
    return True


def test_evaluation_and_report():
    passed = evaluate_session(
        "Why would a Kubernetes pod stay in Pending?",
        "[12:00:00] Why would a Kubernetes pod stay in Pending?",
        "The pod is waiting for a node.",
        None,
        b"png-bytes",
        False,
    )
    assert passed.passed is True
    failed = evaluate_session("Where is the disk?", "unrelated", "", "capture failed", None, True)
    assert failed.passed is False
    assert failed.timed_out is True
    report = render_session_report(
        iteration=2,
        prompt="Why would a Kubernetes pod stay in Pending?",
        transcript="[12:00:00] Why would a Kubernetes pod stay in Pending?",
        ai_text="The pod is waiting for a node.",
        topic_text="Topic: Kubernetes",
        vision_label="agent-hoy training",
        vision_error=None,
        image_name="run-002.png",
        evaluation=passed,
    )
    assert "Training iteration 2" in report
    assert "passed: yes" in report
    assert "run-002.png" in report
    assert "**" not in report
    print("OK training evaluation report")
    return True


def test_training_flag_is_the_rollback():
    assert TRAINING_ENABLED is True
    print("OK training rollback flag")
    return True


def main():
    checks = [
        test_parse_train_command(),
        test_prompts_are_distinct_for_a_seed(),
        test_evaluation_and_report(),
        test_training_flag_is_the_rollback(),
    ]
    if not all(checks):
        print("FAIL training tests")
        return 1
    print("OK all training tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
