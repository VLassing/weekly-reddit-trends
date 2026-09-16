from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from weekly_reddit_trends.stage_b import (
    StageBInput,
)
from weekly_reddit_trends.stage_b_api import (
    HARD_MAX_BATCH_SIZE,
    StageBApiResult,
    classify_stage_b_batch,
)


def make_input(
    post_id: str,
) -> StageBInput:
    return StageBInput(
        post_id=post_id,
        subreddit="TestSubreddit",
        title=f"Test title {post_id}",
        body_excerpt="Test body",
        external_domain=None,
        subreddit_note=None,
    )


def make_parsed_response(
    post_ids: list[str],
):
    classifications = [
        {
            "post_id": post_id,
            "relevance": 2,
            "value": 2,
            "kind": "discussion",
            "skip": False,
            "skip_reason": None,
        }
        for post_id in post_ids
    ]

    from weekly_reddit_trends.stage_b import (
        StageBBatchResponse,
    )

    return StageBBatchResponse(
        classifications=classifications
    )


def make_raw_response(
    post_ids: list[str],
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    reasoning_tokens: int = 0,
):
    parsed = make_parsed_response(
        post_ids
    )

    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(
            input_tokens + output_tokens
        ),
        output_tokens_details=SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        ),
    )

    return SimpleNamespace(
        output_parsed=parsed,
        usage=usage,
    )


class FakeResponses:
    def __init__(
        self,
        outcomes: list[object],
    ) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1

        outcome = self.outcomes[
            self.calls - 1
        ]

        if isinstance(
            outcome,
            Exception,
        ):
            raise outcome

        return outcome


class FakeClient:
    def __init__(
        self,
        outcomes: list[object],
    ) -> None:
        self.responses = FakeResponses(
            outcomes
        )


class StageBApiTest(unittest.TestCase):
    def test_successful_api_batch(self) -> None:
        batch = [
            make_input("a"),
            make_input("b"),
        ]

        client = FakeClient(
            [
                make_raw_response(
                    ["a", "b"],
                    input_tokens=120,
                    output_tokens=40,
                )
            ]
        )

        result = classify_stage_b_batch(
            batch,
            client=client,
            max_attempts=1,
        )

        self.assertIsInstance(
            result,
            StageBApiResult,
        )

        self.assertEqual(
            result.attempts,
            1,
        )

        self.assertEqual(
            result.usage.input_tokens,
            120,
        )

        self.assertEqual(
            result.usage.output_tokens,
            40,
        )

        self.assertEqual(
            result.usage.total_tokens,
            160,
        )

        self.assertEqual(
            result.usage.reasoning_tokens,
            0,
        )

        self.assertEqual(
            client.responses.calls,
            1,
        )

    def test_retry_after_first_failure(
        self,
    ) -> None:
        batch = [
            make_input("a"),
        ]

        client = FakeClient(
            [
                RuntimeError(
                    "Temporary failure"
                ),
                make_raw_response(
                    ["a"]
                ),
            ]
        )

        result = classify_stage_b_batch(
            batch,
            client=client,
            max_attempts=2,
        )

        self.assertEqual(
            result.attempts,
            2,
        )

        self.assertEqual(
            client.responses.calls,
            2,
        )

    def test_failure_after_max_attempts(
        self,
    ) -> None:
        batch = [
            make_input("a"),
        ]

        client = FakeClient(
            [
                RuntimeError(
                    "Failure one"
                ),
                RuntimeError(
                    "Failure two"
                ),
            ]
        )

        with self.assertRaises(
            RuntimeError
        ):
            classify_stage_b_batch(
                batch,
                client=client,
                max_attempts=2,
            )

        self.assertEqual(
            client.responses.calls,
            2,
        )

    def test_rejects_empty_batch(
        self,
    ) -> None:
        client = FakeClient([])

        with self.assertRaises(
            ValueError
        ):
            classify_stage_b_batch(
                [],
                client=client,
            )

    def test_rejects_batch_above_hard_limit(
        self,
    ) -> None:
        batch = [
            make_input(
                f"post_{index}"
            )
            for index in range(
                HARD_MAX_BATCH_SIZE + 1
            )
        ]

        client = FakeClient([])

        with self.assertRaises(
            ValueError
        ):
            classify_stage_b_batch(
                batch,
                client=client,
            )

    def test_rejects_too_many_attempts(
        self,
    ) -> None:
        batch = [
            make_input("a"),
        ]

        client = FakeClient([])

        with self.assertRaises(
            ValueError
        ):
            classify_stage_b_batch(
                batch,
                client=client,
                max_attempts=3,
            )

    def test_rejects_excessive_output_limit(
        self,
    ) -> None:
        batch = [
            make_input("a"),
        ]

        client = FakeClient([])

        with self.assertRaises(
            ValueError
        ):
            classify_stage_b_batch(
                batch,
                client=client,
                max_output_tokens=5001,
            )

    def test_rejects_missing_post_id_in_response(
        self,
    ) -> None:
        batch = [
            make_input("a"),
            make_input("b"),
        ]

        client = FakeClient(
            [
                make_raw_response(
                    ["a"]
                )
            ]
        )

        with self.assertRaises(
            RuntimeError
        ):
            classify_stage_b_batch(
                batch,
                client=client,
                max_attempts=1,
            )

    def test_rejects_no_parsed_output(
        self,
    ) -> None:
        batch = [
            make_input("a"),
        ]

        raw_response = SimpleNamespace(
            output_parsed=None,
            usage=None,
        )

        client = FakeClient(
            [
                raw_response
            ]
        )

        with self.assertRaises(
            RuntimeError
        ):
            classify_stage_b_batch(
                batch,
                client=client,
                max_attempts=1,
            )


if __name__ == "__main__":
    unittest.main()