from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
RANKING_CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "ranking.yaml"
)
COMMUNITIES_CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "communities.yaml"
)

sys.path.insert(0, str(SRC_DIR))

from weekly_reddit_trends.models import Post
from weekly_reddit_trends.stage_a import StageAResult
from weekly_reddit_trends.stage_b import (
    StageBBatchResponse,
    StageBClassification,
    StageBInput,
    batch_stage_b_inputs,
    build_stage_b_inputs,
    calculate_preliminary_score,
    load_stage_b_config,
    load_subreddit_notes,
    score_stage_b_response,
    serialize_stage_b_batch,
    validate_batch_response,
)


def make_post(
    post_id: str,
    subreddit: str = "TestSubreddit",
    body: str = "Test body",
    domain: str | None = None,
    is_self: bool = True,
) -> Post:
    now = datetime(
        2026,
        9,
        10,
        12,
        0,
        tzinfo=timezone.utc,
    )

    return Post(
        id=post_id,
        subreddit=subreddit,
        title=f"Test post {post_id}",
        body=body,
        score=10,
        comment_count=5,
        upvote_ratio=None,
        created_at=now,
        permalink=(
            f"https://www.reddit.com/"
            f"r/{subreddit}/comments/{post_id}/"
        ),
        external_url=(
            None
            if is_self
            else "https://example.com/item"
        ),
        domain=domain,
        post_type="text" if is_self else "link",
        flair=None,
        is_self=is_self,
        is_nsfw=False,
        is_stickied=False,
        is_locked=False,
        is_crosspost=False,
        crosspost_parent=None,
        removed_state="none",
        provider="fixture",
        fetched_at=now,
        score_at_fetch=10,
    )


def make_stage_a_result(
    post_id: str,
    *,
    keep: bool = True,
    engagement_score: float | None = 0.5,
    subreddit: str = "TestSubreddit",
    body: str = "Test body",
    domain: str | None = None,
    is_self: bool = True,
) -> StageAResult:
    return StageAResult(
        post=make_post(
            post_id=post_id,
            subreddit=subreddit,
            body=body,
            domain=domain,
            is_self=is_self,
        ),
        keep=keep,
        reason="normal" if keep else "low_engagement",
        score_rank=0.5 if keep else None,
        comment_count_rank=0.5 if keep else None,
        engagement_score=engagement_score,
    )


class StageBTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_stage_b_config(
            RANKING_CONFIG_PATH
        )

    def test_load_stage_b_config(self) -> None:
        self.assertEqual(
            self.config.batch_size,
            25,
        )
        self.assertEqual(
            self.config.body_excerpt_chars,
            300,
        )
        self.assertAlmostEqual(
            self.config.engagement_weight,
            0.4,
        )
        self.assertAlmostEqual(
            self.config.llm_weight,
            0.6,
        )

    def test_load_subreddit_notes_when_none_exist(
        self,
    ) -> None:
        notes = load_subreddit_notes(
            COMMUNITIES_CONFIG_PATH
        )

        self.assertEqual(
            notes,
            {},
        )

    def test_build_stage_b_inputs(self) -> None:
        long_body = "x" * 500

        results = [
            make_stage_a_result(
                "keep_self",
                body=long_body,
            ),
            make_stage_a_result(
                "drop_me",
                keep=False,
                engagement_score=None,
            ),
            make_stage_a_result(
                "keep_link",
                subreddit="OpenAI",
                body="Link body",
                domain="example.com",
                is_self=False,
            ),
        ]

        notes = {
            "openai": "Example subreddit note."
        }

        inputs = build_stage_b_inputs(
            results,
            self.config,
            notes,
        )

        self.assertEqual(
            len(inputs),
            2,
        )

        first = inputs[0]
        self.assertEqual(
            first.post_id,
            "keep_self",
        )
        self.assertEqual(
            len(first.body_excerpt),
            300,
        )
        self.assertIsNone(
            first.external_domain
        )
        self.assertIsNone(
            first.subreddit_note
        )

        second = inputs[1]
        self.assertEqual(
            second.post_id,
            "keep_link",
        )
        self.assertEqual(
            second.external_domain,
            "example.com",
        )
        self.assertEqual(
            second.subreddit_note,
            "Example subreddit note.",
        )

    def test_kept_post_requires_engagement_score(
        self,
    ) -> None:
        result = make_stage_a_result(
            "missing_score",
            engagement_score=None,
        )

        with self.assertRaises(ValueError):
            build_stage_b_inputs(
                [result],
                self.config,
            )

    def test_batch_stage_b_inputs(self) -> None:
        inputs = [
            StageBInput(
                post_id=f"post_{index}",
                subreddit="Test",
                title="Title",
                body_excerpt="Body",
                external_domain=None,
                subreddit_note=None,
            )
            for index in range(53)
        ]

        batches = batch_stage_b_inputs(
            inputs,
            batch_size=25,
        )

        self.assertEqual(
            [len(batch) for batch in batches],
            [25, 25, 3],
        )

    def test_serialize_stage_b_batch(self) -> None:
        batch = [
            StageBInput(
                post_id="abc123",
                subreddit="OpenAI",
                title="Example title",
                body_excerpt="Example body",
                external_domain="example.com",
                subreddit_note=None,
            )
        ]

        serialized = serialize_stage_b_batch(
            batch
        )

        self.assertIn(
            '"post_id":"abc123"',
            serialized,
        )
        self.assertIn(
            '"external_domain":"example.com"',
            serialized,
        )
        self.assertNotIn(
            "subreddit_note",
            serialized,
        )

    def test_structured_output_validation(self) -> None:
        valid = StageBClassification(
            post_id="abc123",
            relevance=3,
            value=2,
            kind="research",
            skip=False,
            skip_reason=None,
        )

        self.assertEqual(
            valid.relevance,
            3,
        )

        with self.assertRaises(
            ValidationError
        ):
            StageBClassification(
                post_id="abc123",
                relevance=4,
                value=2,
                kind="research",
                skip=False,
                skip_reason=None,
            )

    def test_validate_batch_response(self) -> None:
        response = StageBBatchResponse(
            classifications=[
                StageBClassification(
                    post_id="a",
                    relevance=3,
                    value=2,
                    kind="news",
                    skip=False,
                    skip_reason=None,
                ),
                StageBClassification(
                    post_id="b",
                    relevance=0,
                    value=0,
                    kind="other",
                    skip=True,
                    skip_reason="Clearly off-topic",
                ),
            ]
        )

        validate_batch_response(
            response,
            ["a", "b"],
        )

    def test_validate_batch_response_rejects_missing_id(
        self,
    ) -> None:
        response = StageBBatchResponse(
            classifications=[
                StageBClassification(
                    post_id="a",
                    relevance=2,
                    value=2,
                    kind="discussion",
                    skip=False,
                    skip_reason=None,
                )
            ]
        )

        with self.assertRaises(ValueError):
            validate_batch_response(
                response,
                ["a", "b"],
            )

    def test_validate_batch_response_rejects_bad_skip_reason(
        self,
    ) -> None:
        response = StageBBatchResponse(
            classifications=[
                StageBClassification(
                    post_id="a",
                    relevance=0,
                    value=0,
                    kind="other",
                    skip=True,
                    skip_reason=None,
                )
            ]
        )

        with self.assertRaises(ValueError):
            validate_batch_response(
                response,
                ["a"],
            )

    def test_calculate_preliminary_score(self) -> None:
        score = calculate_preliminary_score(
            engagement_score=0.5,
            relevance=3,
            value=3,
            config=self.config,
        )

        self.assertAlmostEqual(
            score,
            0.8,
        )

    def test_score_stage_b_response(self) -> None:
        response = StageBBatchResponse(
            classifications=[
                StageBClassification(
                    post_id="a",
                    relevance=3,
                    value=3,
                    kind="release",
                    skip=False,
                    skip_reason=None,
                ),
                StageBClassification(
                    post_id="b",
                    relevance=1,
                    value=2,
                    kind="discussion",
                    skip=False,
                    skip_reason=None,
                ),
            ]
        )

        scored = score_stage_b_response(
            response=response,
            engagement_scores={
                "a": 0.5,
                "b": 1.0,
            },
            config=self.config,
        )

        by_id = {
            item.post_id: item
            for item in scored
        }

        self.assertAlmostEqual(
            by_id["a"].preliminary_score,
            0.8,
        )

        self.assertAlmostEqual(
            by_id["b"].preliminary_score,
            0.7,
        )


if __name__ == "__main__":
    unittest.main()