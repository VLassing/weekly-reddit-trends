from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
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
    / "stage_b_full.json"
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

EXPECTED_STAGE_A_SURVIVORS = 5338

sys.path.insert(0, str(SRC_DIR))

from weekly_reddit_trends.models import Post
from weekly_reddit_trends.stage_a import (
    load_stage_a_config,
    run_stage_a,
)
from weekly_reddit_trends.stage_b import (
    batch_stage_b_inputs,
    build_stage_b_inputs,
    load_stage_b_config,
    load_subreddit_notes,
    score_stage_b_response,
)
from weekly_reddit_trends.stage_b_api import (
    DEFAULT_MODEL,
    classify_stage_b_batch,
    create_stage_b_client,
)


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def utc_now_text() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


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


def atomic_write_json(
    path: Path,
    data: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
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

    temp_path.replace(path)


def source_fingerprints() -> dict[str, str]:
    return {
        "normalized_sha256": sha256_file(
            NORMALIZED_PATH
        ),
        "ranking_sha256": sha256_file(
            RANKING_CONFIG_PATH
        ),
        "communities_sha256": sha256_file(
            COMMUNITIES_CONFIG_PATH
        ),
    }


def new_checkpoint(
    fingerprints: dict[str, str],
    total_candidates: int,
    batch_size: int,
) -> dict:
    return {
        "schema_version": 1,
        "run": "2026-09-07",
        "model": DEFAULT_MODEL,
        "created_at": utc_now_text(),
        "updated_at": utc_now_text(),
        "complete": False,
        "source": fingerprints,
        "stage_a_survivors": total_candidates,
        "batch_size": batch_size,
        "posts_completed": 0,
        "batches_completed": 0,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        },
        "results": [],
    }


def load_checkpoint() -> dict | None:
    if not OUTPUT_PATH.exists():
        return None

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_checkpoint(
    checkpoint: dict,
    fingerprints: dict[str, str],
    valid_post_ids: set[str],
    batch_size: int,
) -> None:
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError(
            "Unsupported checkpoint schema_version."
        )

    if checkpoint.get("source") != fingerprints:
        raise RuntimeError(
            "Checkpoint source/config fingerprints do not "
            "match the current files. Refusing to resume."
        )

    if checkpoint.get("stage_a_survivors") != len(
        valid_post_ids
    ):
        raise RuntimeError(
            "Checkpoint Stage A survivor count does not "
            "match the current run."
        )

    if checkpoint.get("batch_size") != batch_size:
        raise RuntimeError(
            "Checkpoint batch size does not match "
            "the current Stage B config."
        )

    results = checkpoint.get("results")

    if not isinstance(results, list):
        raise RuntimeError(
            "Checkpoint results must be a list."
        )

    completed_ids = [
        item.get("post_id")
        for item in results
    ]

    if len(completed_ids) != len(
        set(completed_ids)
    ):
        raise RuntimeError(
            "Checkpoint contains duplicate post IDs."
        )

    unknown_ids = (
        set(completed_ids)
        - valid_post_ids
    )

    if unknown_ids:
        raise RuntimeError(
            "Checkpoint contains post IDs that are not "
            "present in the current Stage B candidates."
        )

    if checkpoint.get(
        "posts_completed"
    ) != len(results):
        raise RuntimeError(
            "Checkpoint posts_completed does not match "
            "the number of stored results."
        )


def scored_result_to_dict(
    item,
    posts_by_id: dict[str, Post],
) -> dict:
    post = posts_by_id[item.post_id]

    return {
        "post_id": item.post_id,
        "subreddit": post.subreddit,
        "title": post.title,
        "engagement_score": item.engagement_score,
        "relevance": item.relevance,
        "value": item.value,
        "kind": item.kind,
        "skip": item.skip,
        "skip_reason": item.skip_reason,
        "preliminary_score": item.preliminary_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable Stage B classification for "
            "the full 2026-09-07 dataset."
        )
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help=(
            "Show current Stage B full-run status "
            "without making API calls."
        ),
    )

    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help=(
            "Process at most this many remaining posts "
            "during this invocation."
        ),
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Process at most this many API batches "
            "during this invocation."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Explicitly allow processing all remaining "
            "Stage B candidates."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_posts is not None:
        if args.max_posts <= 0:
            raise ValueError(
                "--max-posts must be positive."
            )

    if args.max_batches is not None:
        if args.max_batches <= 0:
            raise ValueError(
                "--max-batches must be positive."
            )

    if args.full and (
        args.max_posts is not None
        or args.max_batches is not None
    ):
        raise ValueError(
            "--full cannot be combined with "
            "--max-posts or --max-batches."
        )

    if (
        not args.status
        and not args.full
        and args.max_posts is None
        and args.max_batches is None
    ):
        raise RuntimeError(
            "Refusing to make API calls without an "
            "explicit safety limit or --full. "
            "Use --max-batches N, --max-posts N, "
            "or --full."
        )

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

    if len(kept_results) != (
        EXPECTED_STAGE_A_SURVIVORS
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_STAGE_A_SURVIVORS} "
            "Stage A survivors, but got "
            f"{len(kept_results)}. "
            "Refusing to continue."
        )

    kept_results.sort(
        key=lambda result: (
            result.post.subreddit.casefold(),
            result.post.id,
        )
    )

    stage_b_inputs = build_stage_b_inputs(
        kept_results,
        stage_b_config,
        subreddit_notes,
    )

    posts_by_id = {
        result.post.id: result.post
        for result in kept_results
    }

    engagement_scores = {
        result.post.id: result.engagement_score
        for result in kept_results
        if result.engagement_score is not None
    }

    valid_post_ids = {
        item.post_id
        for item in stage_b_inputs
    }

    fingerprints = source_fingerprints()

    checkpoint = load_checkpoint()

    if checkpoint is None:
        checkpoint = new_checkpoint(
            fingerprints=fingerprints,
            total_candidates=len(
                stage_b_inputs
            ),
            batch_size=(
                stage_b_config.batch_size
            ),
        )
    else:
        validate_checkpoint(
            checkpoint=checkpoint,
            fingerprints=fingerprints,
            valid_post_ids=valid_post_ids,
            batch_size=(
                stage_b_config.batch_size
            ),
        )

    completed_ids = {
        item["post_id"]
        for item in checkpoint["results"]
    }

    pending_inputs = [
        item
        for item in stage_b_inputs
        if item.post_id not in completed_ids
    ]

    print(
        f"Stage A survivors: "
        f"{len(stage_b_inputs)}"
    )
    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )
    print(
        f"Remaining: "
        f"{len(pending_inputs)}"
    )
    print(
        f"Batch size: "
        f"{stage_b_config.batch_size}"
    )
    print(
        f"Checkpoint: {OUTPUT_PATH}"
    )
    print()

    if args.status:
        print("STATUS ONLY - no API calls made.")
        return 0

    if not pending_inputs:
        checkpoint["complete"] = True
        checkpoint["updated_at"] = utc_now_text()

        atomic_write_json(
            OUTPUT_PATH,
            checkpoint,
        )

        print(
            "Stage B full run is already complete."
        )
        return 0

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this "
            "PowerShell session."
        )

    selected_inputs = pending_inputs

    if args.max_posts is not None:
        selected_inputs = selected_inputs[
            :args.max_posts
        ]

    batches = batch_stage_b_inputs(
        selected_inputs,
        stage_b_config.batch_size,
    )

    if args.max_batches is not None:
        batches = batches[
            :args.max_batches
        ]

    if not batches:
        print(
            "No batches selected. No API calls made."
        )
        return 0

    posts_selected = sum(
        len(batch)
        for batch in batches
    )

    print(
        f"This invocation will process: "
        f"{posts_selected} posts"
    )
    print(
        f"This invocation will send: "
        f"{len(batches)} batches"
    )
    print()

    client = create_stage_b_client()

    try:
        for invocation_batch_number, batch in enumerate(
            batches,
            start=1,
        ):
            total_batch_number = (
                checkpoint["batches_completed"]
                + 1
            )

            print(
                f"Batch {invocation_batch_number}/"
                f"{len(batches)} "
                f"(cumulative batch "
                f"{total_batch_number}) - "
                f"{len(batch)} posts"
            )

            api_result = classify_stage_b_batch(
                batch,
                client=client,
            )

            batch_engagement_scores = {
                item.post_id: engagement_scores[
                    item.post_id
                ]
                for item in batch
            }

            scored_results = (
                score_stage_b_response(
                    response=api_result.response,
                    engagement_scores=(
                        batch_engagement_scores
                    ),
                    config=stage_b_config,
                )
            )

            checkpoint["results"].extend(
                scored_result_to_dict(
                    item,
                    posts_by_id,
                )
                for item in scored_results
            )

            checkpoint[
                "posts_completed"
            ] = len(
                checkpoint["results"]
            )

            checkpoint[
                "batches_completed"
            ] += 1

            usage = checkpoint["usage"]

            usage["input_tokens"] += (
                api_result.usage.input_tokens
            )
            usage["output_tokens"] += (
                api_result.usage.output_tokens
            )
            usage["total_tokens"] += (
                api_result.usage.total_tokens
            )
            usage["reasoning_tokens"] += (
                api_result.usage.reasoning_tokens
            )

            checkpoint["updated_at"] = (
                utc_now_text()
            )

            checkpoint["complete"] = (
                checkpoint["posts_completed"]
                == len(stage_b_inputs)
            )

            atomic_write_json(
                OUTPUT_PATH,
                checkpoint,
            )

            print(
                f"  saved checkpoint - "
                f"{checkpoint['posts_completed']}/"
                f"{len(stage_b_inputs)} posts complete"
            )

    except KeyboardInterrupt:
        print()
        print(
            "Interrupted with Ctrl+C. "
            "All previously completed batches "
            "remain saved."
        )
        return 130

    print()
    print("=" * 80)

    remaining = (
        len(stage_b_inputs)
        - checkpoint["posts_completed"]
    )

    if checkpoint["complete"]:
        print("FULL STAGE B RUN COMPLETE")
    else:
        print("STAGE B INVOCATION COMPLETE")

    print(
        f"posts_completed: "
        f"{checkpoint['posts_completed']}"
    )
    print(
        f"posts_remaining: {remaining}"
    )
    print(
        f"batches_completed: "
        f"{checkpoint['batches_completed']}"
    )
    print(
        f"input_tokens: "
        f"{checkpoint['usage']['input_tokens']}"
    )
    print(
        f"output_tokens: "
        f"{checkpoint['usage']['output_tokens']}"
    )
    print(
        f"total_tokens: "
        f"{checkpoint['usage']['total_tokens']}"
    )
    print(
        f"reasoning_tokens: "
        f"{checkpoint['usage']['reasoning_tokens']}"
    )
    print(
        f"saved: {OUTPUT_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())