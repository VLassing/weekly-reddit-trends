from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

NORMALIZED_PATH = (
    REPO_ROOT
    / "runs"
    / "2026-09-07"
    / "normalized.json"
)

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
from weekly_reddit_trends.stage_a import (
    load_stage_a_config,
    run_stage_a,
)
from weekly_reddit_trends.stage_b import (
    build_stage_b_inputs,
    load_stage_b_config,
    load_subreddit_notes,
    serialize_stage_b_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a small deterministic Stage B sample "
            "without calling the OpenAI API."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of Stage A survivors to inspect.",
    )

    return parser.parse_args()


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
        created_at=parse_datetime(
            data["created_at"]
        ),
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
        fetched_at=parse_datetime(
            data["fetched_at"]
        ),
        score_at_fetch=data["score_at_fetch"],
    )


def load_posts() -> list[Post]:
    with NORMALIZED_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    return [
        post_from_dict(item)
        for item in raw["posts"]
    ]


def select_spread_sample(
    results,
    count: int,
):
    survivors = [
        result
        for result in results
        if result.keep
    ]

    survivors.sort(
        key=lambda result: (
            result.engagement_score,
            result.post.subreddit.casefold(),
            result.post.id,
        )
    )

    if count <= 0:
        raise ValueError(
            "--count must be positive."
        )

    if count >= len(survivors):
        return survivors

    if count == 1:
        return [
            survivors[len(survivors) // 2]
        ]

    last_index = len(survivors) - 1

    selected_indexes = [
        round(
            position
            * last_index
            / (count - 1)
        )
        for position in range(count)
    ]

    return [
        survivors[index]
        for index in selected_indexes
    ]


def main() -> int:
    args = parse_args()

    posts = load_posts()

    stage_a_config = load_stage_a_config(
        RANKING_CONFIG_PATH
    )

    stage_b_config = load_stage_b_config(
        RANKING_CONFIG_PATH
    )

    subreddit_notes = load_subreddit_notes(
        COMMUNITIES_CONFIG_PATH
    )

    stage_a_results = run_stage_a(
        posts,
        stage_a_config,
    )

    kept = [
        result
        for result in stage_a_results
        if result.keep
    ]

    sample_results = select_spread_sample(
        stage_a_results,
        args.count,
    )

    sample_inputs = build_stage_b_inputs(
        sample_results,
        stage_b_config,
        subreddit_notes,
    )

    print(
        f"Total normalized posts: {len(posts)}"
    )
    print(
        f"Stage A survivors: {len(kept)}"
    )
    print(
        f"Sample size: {len(sample_results)}"
    )
    print()

    print("SAMPLE")
    print("=" * 80)

    for index, result in enumerate(
        sample_results,
        start=1,
    ):
        post = result.post

        print(
            f"[{index:02d}] "
            f"r/{post.subreddit} | "
            f"{post.id}"
        )
        print(
            "engagement_score:",
            f"{result.engagement_score:.4f}",
        )
        print(
            "score/comments:",
            f"{post.score}/{post.comment_count}",
        )
        print(
            "title:",
            post.title,
        )

        if post.domain:
            print(
                "domain:",
                post.domain,
            )

        if post.body:
            excerpt = (
                post.body
                .replace("\n", " ")
                .strip()
            )

            print(
                "body:",
                excerpt[:300],
            )

        print()

    print("MODEL INPUT")
    print("=" * 80)
    print(
        serialize_stage_b_batch(
            sample_inputs
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())