from __future__ import annotations

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
    score_stage_b_response,
)
from weekly_reddit_trends.stage_b_api import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    classify_stage_b_batch,
)


SAMPLE_SIZE = 12


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
            "Sample size must be positive."
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

    sample_results = select_spread_sample(
        stage_a_results,
        SAMPLE_SIZE,
    )

    sample_inputs = build_stage_b_inputs(
        sample_results,
        stage_b_config,
        subreddit_notes,
    )

    engagement_scores = {
        result.post.id: result.engagement_score
        for result in sample_results
        if result.engagement_score is not None
    }

    posts_by_id = {
        result.post.id: result.post
        for result in sample_results
    }

    print(
        f"Model: {DEFAULT_MODEL}"
    )
    print(
        f"Posts in API batch: {len(sample_inputs)}"
    )
    print(
        f"Max output tokens: "
        f"{DEFAULT_MAX_OUTPUT_TOKENS}"
    )
    print()
    print(
        "Calling Stage B API module..."
    )
    print()

    api_result = classify_stage_b_batch(
        sample_inputs,
        max_attempts=1,
    )

    scored = score_stage_b_response(
        response=api_result.response,
        engagement_scores=engagement_scores,
        config=stage_b_config,
    )

    print("CLASSIFICATIONS")
    print("=" * 80)

    for item in scored:
        post = posts_by_id[item.post_id]

        print(
            f"{item.post_id} | "
            f"r/{post.subreddit}"
        )
        print(
            f"title: {post.title}"
        )
        print(
            f"relevance/value: "
            f"{item.relevance}/{item.value}"
        )
        print(
            f"kind: {item.kind}"
        )
        print(
            f"skip: {item.skip}"
        )
        print(
            f"skip_reason: {item.skip_reason}"
        )
        print(
            f"engagement_score: "
            f"{item.engagement_score:.4f}"
        )
        print(
            f"preliminary_score: "
            f"{item.preliminary_score:.4f}"
        )
        print()

    print("API")
    print("=" * 80)
    print(
        f"attempts: {api_result.attempts}"
    )

    print()
    print("USAGE")
    print("=" * 80)
    print(
        f"input_tokens: "
        f"{api_result.usage.input_tokens}"
    )
    print(
        f"output_tokens: "
        f"{api_result.usage.output_tokens}"
    )
    print(
        f"total_tokens: "
        f"{api_result.usage.total_tokens}"
    )
    print(
        f"reasoning_tokens: "
        f"{api_result.usage.reasoning_tokens}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())