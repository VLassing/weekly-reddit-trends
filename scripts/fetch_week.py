from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CONFIG_PATH = REPO_ROOT / "config" / "communities.yaml"
RUNS_DIR = REPO_ROOT / "runs"

sys.path.insert(0, str(SRC_DIR))

from weekly_reddit_trends.models import Post
from weekly_reddit_trends.sources.arctic import ArcticShiftSource


STOCKHOLM = ZoneInfo("Europe/Stockholm")
EXPECTED_COMMUNITY_COUNT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one complete Weekly Reddit Trends v0 week."
    )
    parser.add_argument(
        "--week-start",
        help=(
            "Optional Monday in YYYY-MM-DD format. "
            "If omitted, fetch the most recently completed Monday-to-Monday week."
        ),
    )
    return parser.parse_args()


def load_communities() -> list[str]:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    communities = [
        item["name"]
        for item in config["profile"]["communities"]
    ]

    if len(communities) != EXPECTED_COMMUNITY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COMMUNITY_COUNT} communities, "
            f"found {len(communities)}."
        )

    return communities


def calculate_window(
    requested_week_start: str | None,
) -> tuple[datetime, datetime, datetime, datetime]:
    if requested_week_start:
        start_date = date.fromisoformat(requested_week_start)

        if start_date.weekday() != 0:
            raise ValueError(
                "--week-start must be a Monday."
            )

        start_local = datetime.combine(
            start_date,
            datetime_time.min,
            tzinfo=STOCKHOLM,
        )
        end_local = start_local + timedelta(days=7)

    else:
        now_local = datetime.now(STOCKHOLM)

        this_monday = (
            now_local.date()
            - timedelta(days=now_local.weekday())
        )

        end_local = datetime.combine(
            this_monday,
            datetime_time.min,
            tzinfo=STOCKHOLM,
        )

        start_local = end_local - timedelta(days=7)

    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    return start_local, end_local, start_utc, end_utc


def iso_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def post_to_dict(post: Post) -> dict:
    data = asdict(post)

    data["created_at"] = iso_utc(post.created_at)
    data["fetched_at"] = iso_utc(post.fetched_at)

    return data


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def main() -> int:
    args = parse_args()

    communities = load_communities()

    (
        start_local,
        end_local,
        start_utc,
        end_utc,
    ) = calculate_window(args.week_start)

    run_dir = RUNS_DIR / start_local.date().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = run_dir / "normalized.json"
    metadata_path = run_dir / "run.json"

    source = ArcticShiftSource()

    all_posts: list[Post] = []
    community_counts: dict[str, int] = {}
    errors: list[dict[str, str]] = []

    run_started_at = datetime.now(timezone.utc)
    timer_started = time.perf_counter()

    print(
        "Window:",
        iso_utc(start_utc),
        "->",
        iso_utc(end_utc),
    )
    print(f"Communities: {len(communities)}")
    print()

    for index, subreddit in enumerate(communities, start=1):
        print(
            f"[{index:02d}/{len(communities)}] "
            f"r/{subreddit} ...",
            end=" ",
            flush=True,
        )

        try:
            posts = source.fetch_posts(
                subreddit,
                start_utc,
                end_utc,
            )

            all_posts.extend(posts)
            community_counts[subreddit] = len(posts)

            print(f"{len(posts)} posts")

        except Exception as exc:
            community_counts[subreddit] = 0

            errors.append(
                {
                    "subreddit": subreddit,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

            print(
                f"ERROR: {type(exc).__name__}: {exc}"
            )

    all_posts.sort(
        key=lambda post: (
            post.subreddit.casefold(),
            post.created_at,
            post.id,
        )
    )

    normalized = {
        "schema_version": 1,
        "provider": source.provider_name,
        "window": {
            "timezone": "Europe/Stockholm",
            "start_local": start_local.isoformat(),
            "end_local": end_local.isoformat(),
            "start_utc": iso_utc(start_utc),
            "end_utc": iso_utc(end_utc),
        },
        "posts": [
            post_to_dict(post)
            for post in all_posts
        ],
    }

    run_finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round(
        time.perf_counter() - timer_started,
        3,
    )

    metadata = {
        "schema_version": 1,
        "provider": source.provider_name,
        "status": "ok" if not errors else "partial",
        "run_started_at": iso_utc(run_started_at),
        "run_finished_at": iso_utc(run_finished_at),
        "elapsed_seconds": elapsed_seconds,
        "window": {
            "timezone": "Europe/Stockholm",
            "start_local": start_local.isoformat(),
            "end_local": end_local.isoformat(),
            "start_utc": iso_utc(start_utc),
            "end_utc": iso_utc(end_utc),
        },
        "community_count": len(communities),
        "community_counts": community_counts,
        "total_posts": len(all_posts),
        "errors": errors,
        "normalized_file": normalized_path.name,
    }

    write_json(normalized_path, normalized)
    write_json(metadata_path, metadata)

    print()
    print(f"Total posts: {len(all_posts)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed_seconds}s")
    print(f"Saved: {normalized_path}")
    print(f"Saved: {metadata_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())