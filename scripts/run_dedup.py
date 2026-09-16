from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

INPUT_PATH = (
    REPO_ROOT
    / "runs"
    / "2026-09-07"
    / "stage_b_shortlist_hydrated.json"
)

OUTPUT_PATH = (
    REPO_ROOT
    / "runs"
    / "2026-09-07"
    / "dedup.json"
)

EXPECTED_INPUT_POSTS = 937

sys.path.insert(0, str(SRC_DIR))

from weekly_reddit_trends.dedup import (
    build_dedup_groups,
    summarize_dedup_groups,
)


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def load_input() -> dict:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Hydrated shortlist not found: "
            f"{INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if (
        data.get("shortlist_count")
        != EXPECTED_INPUT_POSTS
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_INPUT_POSTS} "
            "shortlisted posts, got "
            f"{data.get('shortlist_count')}."
        )

    if (
        data.get("hydrated_count")
        != EXPECTED_INPUT_POSTS
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_INPUT_POSTS} "
            "hydrated posts, got "
            f"{data.get('hydrated_count')}."
        )

    if data.get("missing_count") != 0:
        raise RuntimeError(
            "Hydrated shortlist contains "
            "missing posts."
        )

    items = data.get("items")

    if not isinstance(items, list):
        raise RuntimeError(
            "Hydrated shortlist items "
            "must be a list."
        )

    if len(items) != EXPECTED_INPUT_POSTS:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_INPUT_POSTS} items, "
            f"got {len(items)}."
        )

    for item in items:
        if not isinstance(
            item.get("stage_b"),
            dict,
        ):
            raise RuntimeError(
                "An item is missing Stage B data."
            )

        if not isinstance(
            item.get("post"),
            dict,
        ):
            raise RuntimeError(
                "An item is missing hydrated "
                "post data."
            )

    return data


def main() -> int:
    source = load_input()

    items = source["items"]

    print(
        f"Hydrated shortlist: "
        f"{len(items)}"
    )

    groups = build_dedup_groups(
        items
    )

    summary = summarize_dedup_groups(
        groups
    )

    if (
        summary["input_posts"]
        != EXPECTED_INPUT_POSTS
    ):
        raise RuntimeError(
            "Dedup output does not account "
            "for every input post."
        )

    output = {
        "schema_version": 1,
        "source_week": (
            source.get(
                "source_week",
                "2026-09-07",
            )
        ),
        "created_at": utc_now_text(),
        "input_artifact": (
            "stage_b_shortlist_hydrated.json"
        ),
        "summary": summary,
        "groups": groups,
    }

    atomic_write_json(
        OUTPUT_PATH,
        output,
    )

    print()
    print("DEDUP COMPLETE")
    print(
        f"Input posts: "
        f"{summary['input_posts']}"
    )
    print(
        f"Groups: "
        f"{summary['groups']}"
    )
    print(
        f"Standalone groups: "
        f"{summary['standalone_groups']}"
    )
    print(
        f"Duplicate groups: "
        f"{summary['duplicate_groups']}"
    )
    print(
        "Posts merged as duplicates: "
        f"{summary['posts_merged_as_duplicates']}"
    )
    print(
        f"Largest group size: "
        f"{summary['largest_group_size']}"
    )
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    if summary["duplicate_groups"]:
        print()
        print(
            "Largest duplicate groups:"
        )

        duplicate_groups = [
            group
            for group in groups
            if group["is_duplicate_group"]
        ]

        duplicate_groups.sort(
            key=lambda group: (
                -group["size"],
                -float(
                    group[
                        "representative"
                    ][
                        "stage_b"
                    ][
                        "preliminary_score"
                    ]
                ),
                group[
                    "representative_post_id"
                ],
            )
        )

        for group in (
            duplicate_groups[:10]
        ):
            representative = (
                group["representative"]
            )

            stage_b = (
                representative["stage_b"]
            )

            print(
                f"  {group['size']} posts | "
                f"{stage_b['preliminary_score']:.3f} | "
                f"r/{stage_b['subreddit']} | "
                f"{stage_b['title'][:100]}"
            )

            if group[
                "duplicate_signals"
            ]:
                print(
                    "    signals: "
                    + ", ".join(
                        group[
                            "duplicate_signals"
                        ]
                    )
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())