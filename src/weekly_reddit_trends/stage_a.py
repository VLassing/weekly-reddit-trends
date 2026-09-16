from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import Post


@dataclass(frozen=True, slots=True)
class StageAConfig:
    removed_states: frozenset[str]
    obvious_spam: bool
    giveaways: bool
    routine_stickies_or_megathreads: bool
    score_weight: float
    comment_count_weight: float
    min_score: int
    min_comment_count: int


@dataclass(frozen=True, slots=True)
class StageAResult:
    post: Post
    keep: bool
    reason: str
    score_rank: float | None = None
    comment_count_rank: float | None = None
    engagement_score: float | None = None


def load_stage_a_config(path: Path) -> StageAConfig:
    with path.open("r", encoding="utf-8-sig") as file:
        raw = yaml.safe_load(file)

    stage_a = raw["stage_a"]
    hard_exclude = stage_a["hard_exclude"]
    engagement = stage_a["engagement"]
    signals = engagement["signals"]
    floor = engagement["absolute_floor"]

    if engagement["normalization"] != "rank_within_subreddit":
        raise ValueError(
            "Stage A currently supports only "
            "'rank_within_subreddit' normalization."
        )

    if floor["logic"] != "any":
        raise ValueError(
            "Stage A currently supports only "
            "'any' absolute-floor logic."
        )

    score_weight = float(signals["score_weight"])
    comment_count_weight = float(signals["comment_count_weight"])

    if score_weight < 0 or comment_count_weight < 0:
        raise ValueError("Engagement weights cannot be negative.")

    if score_weight + comment_count_weight <= 0:
        raise ValueError(
            "At least one engagement weight must be positive."
        )

    return StageAConfig(
        removed_states=frozenset(
            str(value)
            for value in hard_exclude["removed_states"]
        ),
        obvious_spam=bool(hard_exclude["obvious_spam"]),
        giveaways=bool(hard_exclude["giveaways"]),
        routine_stickies_or_megathreads=bool(
            hard_exclude["routine_stickies_or_megathreads"]
        ),
        score_weight=score_weight,
        comment_count_weight=comment_count_weight,
        min_score=int(floor["min_score"]),
        min_comment_count=int(floor["min_comment_count"]),
    )


_GIVEAWAY_RE = re.compile(
    r"\b(?:giveaway|give\s+away|sweepstakes)\b",
    re.IGNORECASE,
)

_ROUTINE_THREAD_PATTERNS = (
    re.compile(
        r"\b(?:daily|weekly|monthly|quarterly)\b"
        r".{0,50}\b(?:thread|megathread|discussion|questions?|help|"
        r"hiring|jobs?|showcase|promotion|self[- ]promo)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:thread|megathread)\b"
        r".{0,50}\b(?:daily|weekly|monthly|quarterly)\b",
        re.IGNORECASE,
    ),
)

_SPAM_PATTERNS = (
    re.compile(
        r"\bbuy\s+(?:reddit\s+)?"
        r"(?:upvotes?|followers?|subscribers?|likes?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bguaranteed\s+"
        r"(?:traffic|followers?|subscribers?|ranking|rankings)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:telegram|whatsapp)\b"
        r".{0,50}\b(?:dm|message|contact)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dm|message|contact)\b"
        r".{0,50}\b(?:telegram|whatsapp)\b",
        re.IGNORECASE,
    ),
)


def _combined_text(post: Post) -> str:
    return f"{post.title}\n{post.body}".strip()


def _is_giveaway(post: Post) -> bool:
    return bool(_GIVEAWAY_RE.search(_combined_text(post)))


def _is_obvious_spam(post: Post) -> bool:
    text = _combined_text(post)

    return any(
        pattern.search(text)
        for pattern in _SPAM_PATTERNS
    )


def _is_routine_sticky_or_megathread(post: Post) -> bool:
    title = post.title.strip()

    if any(
        pattern.search(title)
        for pattern in _ROUTINE_THREAD_PATTERNS
    ):
        return True

    if post.is_stickied and re.search(
        r"\b(?:megathread|hiring thread|jobs? thread|"
        r"questions? thread|help thread|self[- ]promo)\b",
        title,
        re.IGNORECASE,
    ):
        return True

    return False


def _hard_exclusion_reason(
    post: Post,
    config: StageAConfig,
) -> str | None:
    if post.removed_state in config.removed_states:
        return post.removed_state

    if (
        config.giveaways
        and _is_giveaway(post)
    ):
        return "giveaway"

    if (
        config.obvious_spam
        and _is_obvious_spam(post)
    ):
        return "obvious_spam"

    if (
        config.routine_stickies_or_megathreads
        and _is_routine_sticky_or_megathread(post)
    ):
        return "routine_sticky"

    return None


def _passes_engagement_floor(
    post: Post,
    config: StageAConfig,
) -> bool:
    return (
        post.score >= config.min_score
        or post.comment_count >= config.min_comment_count
    )


def _rank_normalize(values: list[int]) -> list[float]:
    if not values:
        return []

    if len(values) == 1:
        return [1.0]

    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    normalized = [0.0] * len(values)
    denominator = len(values) - 1

    start = 0

    while start < len(indexed):
        end = start + 1
        value = indexed[start][1]

        while (
            end < len(indexed)
            and indexed[end][1] == value
        ):
            end += 1

        average_rank = (
            start + (end - 1)
        ) / 2

        rank_value = average_rank / denominator

        for position in range(start, end):
            original_index = indexed[position][0]
            normalized[original_index] = rank_value

        start = end

    return normalized


def _allowed_reason(post: Post) -> str:
    if post.is_locked:
        return "locked_allowed"

    if post.is_crosspost:
        return "crosspost_allowed"

    return "normal"


def run_stage_a(
    posts: list[Post],
    config: StageAConfig,
) -> list[StageAResult]:
    results: list[StageAResult | None] = [None] * len(posts)

    surviving_by_subreddit: dict[
        str,
        list[tuple[int, Post]],
    ] = defaultdict(list)

    for index, post in enumerate(posts):
        exclusion_reason = _hard_exclusion_reason(
            post,
            config,
        )

        if exclusion_reason is not None:
            results[index] = StageAResult(
                post=post,
                keep=False,
                reason=exclusion_reason,
            )
            continue

        if not _passes_engagement_floor(post, config):
            results[index] = StageAResult(
                post=post,
                keep=False,
                reason="low_engagement",
            )
            continue

        surviving_by_subreddit[
            post.subreddit.casefold()
        ].append((index, post))

    weight_total = (
        config.score_weight
        + config.comment_count_weight
    )

    for subreddit_posts in surviving_by_subreddit.values():
        scores = [
            post.score
            for _, post in subreddit_posts
        ]
        comment_counts = [
            post.comment_count
            for _, post in subreddit_posts
        ]

        score_ranks = _rank_normalize(scores)
        comment_ranks = _rank_normalize(comment_counts)

        for position, (index, post) in enumerate(
            subreddit_posts
        ):
            engagement_score = (
                (
                    score_ranks[position]
                    * config.score_weight
                )
                + (
                    comment_ranks[position]
                    * config.comment_count_weight
                )
            ) / weight_total

            results[index] = StageAResult(
                post=post,
                keep=True,
                reason=_allowed_reason(post),
                score_rank=score_ranks[position],
                comment_count_rank=comment_ranks[position],
                engagement_score=engagement_score,
            )

    if any(result is None for result in results):
        raise RuntimeError(
            "Stage A failed to produce a result for every post."
        )

    return [
        result
        for result in results
        if result is not None
    ]