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

OUTPUT_PATH = (
    REPO_ROOT
    / "runs"
    / "2026-09-07"
    / "stage_b_sample.json"
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
from weekly_reddit_trends.stage_b_runner import (
    run_stage_b_batches,
)


SAMPLE_SIZE = 200
MAX_BATCHES = 8


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


def write_json(
    path: Path,
    data: object,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


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

    kept_results = [
        result
        for result in stage_a_results
        if result.keep
    ]

    sample_results = select_spread_sample(
        stage_a_results,
        SAMPLE_SIZE,
    )

    sample_inputs = build_stage_b_inputs(
        sample_results,
        stage_b_config,
        subreddit_notes,
    )

    print(
        f"Stage A survivors: {len(kept_results)}"
    )
    print(
        f"Stage B sample: {len(sample_inputs)}"
    )
    print(
        f"Batch size: {stage_b_config.batch_size}"
    )
    print(
        f"Max batches: {MAX_BATCHES}"
    )
    print()

    run_result = run_stage_b_batches(
        sample_inputs,
        batch_size=stage_b_config.batch_size,
        max_posts=SAMPLE_SIZE,
        max_batches=MAX_BATCHES,
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

    all_scored = []

    for response in run_result.responses:
        scored = score_stage_b_response(
            response=response,
            engagement_scores={
                classification.post_id: engagement_scores[
                    classification.post_id
                ]
                for classification in response.classifications
            },
            config=stage_b_config,
        )

        all_scored.extend(
            scored
        )

    output = {
        "schema_version": 1,
        "sample_size_requested": SAMPLE_SIZE,
        "posts_processed": (
            run_result.posts_processed
        ),
        "batches_processed": (
            run_result.batches_processed
        ),
        "truncated": run_result.truncated,
        "usage": {
            "input_tokens": (
                run_result.input_tokens
            ),
            "output_tokens": (
                run_result.output_tokens
            ),
            "total_tokens": (
                run_result.total_tokens
            ),
            "reasoning_tokens": (
                run_result.reasoning_tokens
            ),
        },
        "results": [
            {
                "post_id": item.post_id,
                "subreddit": (
                    posts_by_id[
                        item.post_id
                    ].subreddit
                ),
                "title": (
                    posts_by_id[
                        item.post_id
                    ].title
                ),
                "engagement_score": (
                    item.engagement_score
                ),
                "relevance": item.relevance,
                "value": item.value,
                "kind": item.kind,
                "skip": item.skip,
                "skip_reason": (
                    item.skip_reason
                ),
                "preliminary_score": (
                    item.preliminary_score
                ),
            }
            for item in all_scored
        ],
    }

    write_json(
        OUTPUT_PATH,
        output,
    )

    skip_count = sum(
        1
        for item in all_scored
        if item.skip
    )

    print()
    print("RUN COMPLETE")
    print("=" * 80)
    print(
        f"posts_processed: "
        f"{run_result.posts_processed}"
    )
    print(
        f"batches_processed: "
        f"{run_result.batches_processed}"
    )
    print(
        f"skipped: {skip_count}"
    )
    print(
        f"input_tokens: "
        f"{run_result.input_tokens}"
    )
    print(
        f"output_tokens: "
        f"{run_result.output_tokens}"
    )
    print(
        f"total_tokens: "
        f"{run_result.total_tokens}"
    )
    print(
        f"reasoning_tokens: "
        f"{run_result.reasoning_tokens}"
    )
    print(
        f"saved: {OUTPUT_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())