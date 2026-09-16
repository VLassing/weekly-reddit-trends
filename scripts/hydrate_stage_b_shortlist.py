from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

STAGE_B_PATH = (
    REPO_ROOT
    / "runs"
    / "2026-09-07"
    / "stage_b_full.json"
)

OUTPUT_PATH = (
    REPO_ROOT
    / "runs"
    / "2026-09-07"
    / "stage_b_shortlist_hydrated.json"
)

EXPECTED_STAGE_B_POSTS = 5338
EXPECTED_SHORTLIST_COUNT = 937

PRIMARY_SCORE_THRESHOLD = 0.80
R3_V3_RESCUE_THRESHOLD = 0.75

sys.path.insert(0, str(SRC_DIR))

from weekly_reddit_trends.models import Post
from weekly_reddit_trends.sources.arctic import (
    ArcticShiftSource,
)


def iso_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def utc_now_text() -> str:
    return iso_utc(
        datetime.now(timezone.utc)
    )


def post_to_dict(post: Post) -> dict:
    data = asdict(post)

    data["created_at"] = iso_utc(
        post.created_at
    )

    data["fetched_at"] = iso_utc(
        post.fetched_at
    )

    return data


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


def load_stage_b() -> dict:
    if not STAGE_B_PATH.exists():
        raise FileNotFoundError(
            f"Stage B file not found: "
            f"{STAGE_B_PATH}"
        )

    with STAGE_B_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not data.get("complete"):
        raise RuntimeError(
            "Stage B full run is not marked complete."
        )

    if (
        data.get("posts_completed")
        != EXPECTED_STAGE_B_POSTS
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_STAGE_B_POSTS} "
            "completed Stage B posts, got "
            f"{data.get('posts_completed')}."
        )

    results = data.get("results")

    if not isinstance(results, list):
        raise RuntimeError(
            "Stage B results must be a list."
        )

    if len(results) != EXPECTED_STAGE_B_POSTS:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_STAGE_B_POSTS} "
            "Stage B results, got "
            f"{len(results)}."
        )

    post_ids = [
        item["post_id"]
        for item in results
    ]

    if len(post_ids) != len(
        set(post_ids)
    ):
        raise RuntimeError(
            "Stage B results contain duplicate "
            "post IDs."
        )

    return data


def is_shortlisted(
    item: dict,
) -> bool:
    if item["skip"]:
        return False

    score = item["preliminary_score"]

    if score >= PRIMARY_SCORE_THRESHOLD:
        return True

    return (
        item["relevance"] == 3
        and item["value"] == 3
        and score >= R3_V3_RESCUE_THRESHOLD
    )


def build_shortlist(
    stage_b: dict,
) -> list[dict]:
    shortlist = [
        item
        for item in stage_b["results"]
        if is_shortlisted(item)
    ]

    shortlist.sort(
        key=lambda item: (
            -item["preliminary_score"],
            item["subreddit"].casefold(),
            item["post_id"],
        )
    )

    if len(shortlist) != (
        EXPECTED_SHORTLIST_COUNT
    ):
        raise RuntimeError(
            "Expected shortlist size "
            f"{EXPECTED_SHORTLIST_COUNT}, "
            f"got {len(shortlist)}. "
            "Refusing to hydrate because the "
            "frozen shortlist rule no longer "
            "matches the verified result."
        )

    return shortlist


def main() -> int:
    stage_b = load_stage_b()

    shortlist = build_shortlist(
        stage_b
    )

    post_ids = [
        item["post_id"]
        for item in shortlist
    ]

    print(
        f"Stage B results: "
        f"{len(stage_b['results'])}"
    )

    print(
        f"Shortlist: "
        f"{len(shortlist)}"
    )

    print(
        "Rule: score >= 0.80 OR "
        "(R3/V3 AND score >= 0.75)"
    )

    estimated_requests = (
        len(post_ids) + 499
    ) // 500

    print(
        f"Expected Arctic requests: "
        f"{estimated_requests}"
    )

    print()

    source = ArcticShiftSource()

    started = time.perf_counter()

    hydrated_posts = (
        source.fetch_posts_by_ids(
            post_ids
        )
    )

    elapsed_seconds = round(
        time.perf_counter() - started,
        3,
    )

    posts_by_id = {
        post.id: post
        for post in hydrated_posts
    }

    hydrated_ids = set(
        posts_by_id
    )

    requested_ids = set(
        post_ids
    )

    unexpected_ids = (
        hydrated_ids
        - requested_ids
    )

    if unexpected_ids:
        raise RuntimeError(
            "Arctic returned unexpected post IDs: "
            + ", ".join(
                sorted(unexpected_ids)
            )
        )

    missing_ids = [
        post_id
        for post_id in post_ids
        if post_id not in hydrated_ids
    ]

    items = [
        {
            "stage_b": item,
            "post": (
                post_to_dict(
                    posts_by_id[
                        item["post_id"]
                    ]
                )
                if item["post_id"]
                in posts_by_id
                else None
            ),
        }
        for item in shortlist
    ]

    output = {
        "schema_version": 1,
        "source_week": "2026-09-07",
        "provider": source.provider_name,
        "hydrated_at": utc_now_text(),
        "shortlist_rule": {
            "skip_must_be_false": True,
            "primary": {
                "preliminary_score_gte": (
                    PRIMARY_SCORE_THRESHOLD
                ),
            },
            "rescue": {
                "relevance": 3,
                "value": 3,
                "preliminary_score_gte": (
                    R3_V3_RESCUE_THRESHOLD
                ),
            },
        },
        "stage_b_posts": len(
            stage_b["results"]
        ),
        "shortlist_count": len(
            shortlist
        ),
        "hydrated_count": len(
            hydrated_posts
        ),
        "missing_count": len(
            missing_ids
        ),
        "missing_post_ids": (
            missing_ids
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "items": items,
    }

    atomic_write_json(
        OUTPUT_PATH,
        output,
    )

    print(
        f"Hydrated: "
        f"{len(hydrated_posts)}"
        f"/{len(shortlist)}"
    )

    print(
        f"Missing: "
        f"{len(missing_ids)}"
    )

    print(
        f"Elapsed: "
        f"{elapsed_seconds}s"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    if missing_ids:
        print()
        print(
            "Hydration is incomplete. "
            "Output was saved for diagnosis, "
            "but dedup must not start yet."
        )

        print(
            "Missing post IDs:"
        )

        for post_id in missing_ids:
            print(
                f"  {post_id}"
            )

        return 1

    print()
    print(
        "HYDRATION COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())