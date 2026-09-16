from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "stage_a_posts.json"
)
RANKING_CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "ranking.yaml"
)

sys.path.insert(0, str(SRC_DIR))

from weekly_reddit_trends.models import Post
from weekly_reddit_trends.stage_a import (
    load_stage_a_config,
    run_stage_a,
)


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def post_from_dict(data: dict) -> Post:
    return Post(
        id=data["id"],
        subreddit=data["subreddit"],
        title=data["title"],
        body=data["body"],
        score=data["score"],
        comment_count=data["comment_count"],
        upvote_ratio=data["upvote_ratio"],
        created_at=parse_datetime(data["created_at"]),
        permalink=data["permalink"],
        external_url=data["external_url"],
        domain=data["domain"],
        post_type=data["post_type"],
        flair=data["flair"],
        is_self=data["is_self"],
        is_nsfw=data["is_nsfw"],
        is_stickied=data["is_stickied"],
        is_locked=data["is_locked"],
        is_crosspost=data["is_crosspost"],
        crosspost_parent=data["crosspost_parent"],
        removed_state=data["removed_state"],
        provider=data["provider"],
        fetched_at=parse_datetime(data["fetched_at"]),
        score_at_fetch=data["score_at_fetch"],
    )


def make_post(
    post_id: str,
    subreddit: str,
    score: int,
    comment_count: int,
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
        body="Test body",
        score=score,
        comment_count=comment_count,
        upvote_ratio=None,
        created_at=now,
        permalink=(
            f"https://www.reddit.com/"
            f"r/{subreddit}/comments/{post_id}/"
        ),
        external_url=None,
        domain=None,
        post_type="text",
        flair=None,
        is_self=True,
        is_nsfw=False,
        is_stickied=False,
        is_locked=False,
        is_crosspost=False,
        crosspost_parent=None,
        removed_state="none",
        provider="fixture",
        fetched_at=now,
        score_at_fetch=score,
    )


class StageAFixtureTest(unittest.TestCase):
    def test_stage_a_fixture_expectations(self) -> None:
        with FIXTURE_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            fixture = json.load(file)

        posts = [
            post_from_dict(data)
            for data in fixture["posts"]
        ]

        config = load_stage_a_config(
            RANKING_CONFIG_PATH
        )

        results = run_stage_a(
            posts,
            config,
        )

        results_by_id = {
            result.post.id: result
            for result in results
        }

        self.assertEqual(
            set(results_by_id),
            set(fixture["expected"]),
        )

        for post_id, expected in fixture["expected"].items():
            result = results_by_id[post_id]

            with self.subTest(post_id=post_id):
                self.assertEqual(
                    result.keep,
                    expected["keep"],
                )

                self.assertEqual(
                    result.reason,
                    expected["reason"],
                )

                if result.keep:
                    self.assertIsNotNone(
                        result.engagement_score
                    )
                    self.assertGreaterEqual(
                        result.engagement_score,
                        0.0,
                    )
                    self.assertLessEqual(
                        result.engagement_score,
                        1.0,
                    )

                else:
                    self.assertIsNone(
                        result.engagement_score
                    )

    def test_rank_normalized_engagement_within_subreddit(
        self,
    ) -> None:
        posts = [
            make_post(
                "post_a",
                "TestSubreddit",
                score=10,
                comment_count=10,
            ),
            make_post(
                "post_b",
                "TestSubreddit",
                score=20,
                comment_count=30,
            ),
            make_post(
                "post_c",
                "TestSubreddit",
                score=30,
                comment_count=20,
            ),
        ]

        config = load_stage_a_config(
            RANKING_CONFIG_PATH
        )

        results = run_stage_a(
            posts,
            config,
        )

        results_by_id = {
            result.post.id: result
            for result in results
        }

        post_a = results_by_id["post_a"]
        post_b = results_by_id["post_b"]
        post_c = results_by_id["post_c"]

        self.assertAlmostEqual(
            post_a.score_rank,
            0.0,
        )
        self.assertAlmostEqual(
            post_a.comment_count_rank,
            0.0,
        )
        self.assertAlmostEqual(
            post_a.engagement_score,
            0.0,
        )

        self.assertAlmostEqual(
            post_b.score_rank,
            0.5,
        )
        self.assertAlmostEqual(
            post_b.comment_count_rank,
            1.0,
        )
        self.assertAlmostEqual(
            post_b.engagement_score,
            0.75,
        )

        self.assertAlmostEqual(
            post_c.score_rank,
            1.0,
        )
        self.assertAlmostEqual(
            post_c.comment_count_rank,
            0.5,
        )
        self.assertAlmostEqual(
            post_c.engagement_score,
            0.75,
        )


if __name__ == "__main__":
    unittest.main()