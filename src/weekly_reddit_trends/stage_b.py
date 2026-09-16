from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .stage_a import StageAResult


StageBKind = Literal[
    "news",
    "tool",
    "release",
    "discussion",
    "recommendation",
    "comparison",
    "experience",
    "showcase",
    "question",
    "research",
    "other",
]

ALLOWED_KINDS: tuple[str, ...] = (
    "news",
    "tool",
    "release",
    "discussion",
    "recommendation",
    "comparison",
    "experience",
    "showcase",
    "question",
    "research",
    "other",
)


STAGE_B_INSTRUCTIONS = """
You are classifying Reddit posts for a short weekly AI & Tech discovery
report.

The report intentionally covers a broader technology and creative-tech
profile, not only posts explicitly about artificial intelligence.

Relevant subject areas include, among others:
- AI, LLMs, agents, machine learning, and generative AI
- software development, coding tools, and developer workflows
- web development, Webflow, Astro, no-code, and related tooling
- UX, UI, Figma, product design, and digital design workflows
- Blender, 3D, After Effects, motion graphics, and creative software
- SEO and technical SEO
- noteworthy tools, releases, workflow changes, user experiences,
  technical problems, and discussions within those fields

A post does NOT need to mention AI to be relevant.

Use the subreddit as important context. A Blender post should be judged
as a Blender/3D post, a Webflow post as a web-development post, and so
on.

However, inclusion in one of these communities does not automatically
make a post valuable. Routine troubleshooting, generic showcases,
simple beginner questions, low-information posts, or highly personal
content may still receive low relevance or value when they are unlikely
to contribute meaningfully to a concise weekly report.

Evaluate each post independently.

relevance:
0 = outside the report's covered domains or effectively irrelevant
1 = within scope but weak, routine, narrow, or peripheral
2 = clearly relevant and potentially useful to the weekly report
3 = strongly relevant, notable, timely, or broadly interesting

value:
0 = little or no useful information
1 = limited, routine, or narrowly useful
2 = substantive, useful, actionable, or discussion-worthy
3 = unusually informative, novel, consequential, or decision-useful

kind must be one of the allowed categories in the response schema.

Use skip=true only when the post is clearly unsuitable for downstream
analysis despite having passed the deterministic prefilter. Be
conservative about skipping.

Examples of reasons to skip can include:
- genuinely off-topic for the covered domains
- essentially no usable information
- obvious low-value promotion not caught earlier
- content too thin or context-free to contribute to a weekly finding

Do not skip a post merely because it is not about AI.

Do not use engagement numbers as a proxy for relevance or value.
Do not invent information that is not present in the supplied post data.

Return exactly one classification for every supplied post_id.
Do not add or alter post IDs.
""".strip()


@dataclass(frozen=True, slots=True)
class StageBConfig:
    batch_size: int
    body_excerpt_chars: int
    relevance_min: int
    relevance_max: int
    value_min: int
    value_max: int
    kinds: tuple[str, ...]
    engagement_weight: float
    llm_weight: float


@dataclass(frozen=True, slots=True)
class StageBInput:
    post_id: str
    subreddit: str
    title: str
    body_excerpt: str
    external_domain: str | None
    subreddit_note: str | None


@dataclass(frozen=True, slots=True)
class StageBScoredResult:
    post_id: str
    engagement_score: float
    relevance: int
    value: int
    kind: StageBKind
    skip: bool
    skip_reason: str | None
    preliminary_score: float


class StageBClassification(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    post_id: str
    relevance: int = Field(
        ge=0,
        le=3,
    )
    value: int = Field(
        ge=0,
        le=3,
    )
    kind: StageBKind
    skip: bool
    skip_reason: str | None


class StageBBatchResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    classifications: list[StageBClassification]


def load_stage_b_config(path: Path) -> StageBConfig:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    stage_b = raw["stage_b"]
    classifier = stage_b["classifier"]
    preliminary = stage_b["preliminary_score"]

    batch_size = int(classifier["batch_size"])
    body_excerpt_chars = int(
        classifier["body_excerpt_chars"]
    )

    relevance_min = int(
        classifier["relevance"]["min"]
    )
    relevance_max = int(
        classifier["relevance"]["max"]
    )

    value_min = int(
        classifier["value"]["min"]
    )
    value_max = int(
        classifier["value"]["max"]
    )

    kinds = tuple(
        str(value)
        for value in classifier["kinds"]
    )

    engagement_weight = float(
        preliminary["engagement_weight"]
    )
    llm_weight = float(
        preliminary["llm_weight"]
    )

    if batch_size <= 0:
        raise ValueError(
            "Stage B batch_size must be positive."
        )

    if body_excerpt_chars <= 0:
        raise ValueError(
            "Stage B body_excerpt_chars must be positive."
        )

    if (
        relevance_min != 0
        or relevance_max != 3
        or value_min != 0
        or value_max != 3
    ):
        raise ValueError(
            "Stage B v0 currently requires relevance "
            "and value ranges of 0-3."
        )

    if kinds != ALLOWED_KINDS:
        raise ValueError(
            "Stage B kinds in ranking.yaml do not match "
            "the v0 Structured Output schema."
        )

    if engagement_weight < 0 or llm_weight < 0:
        raise ValueError(
            "Stage B preliminary-score weights "
            "cannot be negative."
        )

    if engagement_weight + llm_weight <= 0:
        raise ValueError(
            "At least one Stage B preliminary-score "
            "weight must be positive."
        )

    return StageBConfig(
        batch_size=batch_size,
        body_excerpt_chars=body_excerpt_chars,
        relevance_min=relevance_min,
        relevance_max=relevance_max,
        value_min=value_min,
        value_max=value_max,
        kinds=kinds,
        engagement_weight=engagement_weight,
        llm_weight=llm_weight,
    )


def load_subreddit_notes(
    path: Path,
) -> dict[str, str]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    notes: dict[str, str] = {}

    for community in raw["profile"]["communities"]:
        note = community.get("note")

        if note is None:
            continue

        note_text = str(note).strip()

        if not note_text:
            continue

        notes[
            str(community["name"]).casefold()
        ] = note_text

    return notes


def _body_excerpt(
    body: str,
    max_chars: int,
) -> str:
    return body.strip()[:max_chars]


def build_stage_b_inputs(
    stage_a_results: list[StageAResult],
    config: StageBConfig,
    subreddit_notes: dict[str, str] | None = None,
) -> list[StageBInput]:
    notes = subreddit_notes or {}

    inputs: list[StageBInput] = []

    for result in stage_a_results:
        if not result.keep:
            continue

        if result.engagement_score is None:
            raise ValueError(
                f"Kept Stage A post {result.post.id} "
                "has no engagement_score."
            )

        post = result.post

        external_domain = (
            None
            if post.is_self
            else post.domain
        )

        inputs.append(
            StageBInput(
                post_id=post.id,
                subreddit=post.subreddit,
                title=post.title.strip(),
                body_excerpt=_body_excerpt(
                    post.body,
                    config.body_excerpt_chars,
                ),
                external_domain=external_domain,
                subreddit_note=notes.get(
                    post.subreddit.casefold()
                ),
            )
        )

    return inputs


def batch_stage_b_inputs(
    inputs: list[StageBInput],
    batch_size: int,
) -> list[list[StageBInput]]:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    return [
        inputs[index:index + batch_size]
        for index in range(
            0,
            len(inputs),
            batch_size,
        )
    ]


def stage_b_input_to_dict(
    item: StageBInput,
) -> dict[str, object]:
    data: dict[str, object] = {
        "post_id": item.post_id,
        "subreddit": item.subreddit,
        "title": item.title,
        "body_excerpt": item.body_excerpt,
    }

    if item.external_domain is not None:
        data["external_domain"] = (
            item.external_domain
        )

    if item.subreddit_note is not None:
        data["subreddit_note"] = (
            item.subreddit_note
        )

    return data


def serialize_stage_b_batch(
    batch: list[StageBInput],
) -> str:
    payload = {
        "posts": [
            stage_b_input_to_dict(item)
            for item in batch
        ]
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_batch_response(
    response: StageBBatchResponse,
    expected_post_ids: list[str],
) -> None:
    actual_post_ids = [
        item.post_id
        for item in response.classifications
    ]

    if len(actual_post_ids) != len(
        set(actual_post_ids)
    ):
        raise ValueError(
            "Stage B response contains duplicate post IDs."
        )

    expected = set(expected_post_ids)
    actual = set(actual_post_ids)

    missing = expected - actual
    unexpected = actual - expected

    if missing or unexpected:
        details: list[str] = []

        if missing:
            details.append(
                "missing="
                + ",".join(sorted(missing))
            )

        if unexpected:
            details.append(
                "unexpected="
                + ",".join(sorted(unexpected))
            )

        raise ValueError(
            "Stage B response post IDs do not match "
            "the request: "
            + "; ".join(details)
        )

    for item in response.classifications:
        if item.skip:
            if (
                item.skip_reason is None
                or not item.skip_reason.strip()
            ):
                raise ValueError(
                    f"Skipped post {item.post_id} "
                    "must include skip_reason."
                )
        elif item.skip_reason is not None:
            raise ValueError(
                f"Non-skipped post {item.post_id} "
                "must have skip_reason=null."
            )


def calculate_preliminary_score(
    engagement_score: float,
    relevance: int,
    value: int,
    config: StageBConfig,
) -> float:
    if not 0.0 <= engagement_score <= 1.0:
        raise ValueError(
            "engagement_score must be between 0 and 1."
        )

    if not (
        config.relevance_min
        <= relevance
        <= config.relevance_max
    ):
        raise ValueError(
            "relevance is outside the configured range."
        )

    if not (
        config.value_min
        <= value
        <= config.value_max
    ):
        raise ValueError(
            "value is outside the configured range."
        )

    llm_score = (
        relevance + value
    ) / 6.0

    weight_total = (
        config.engagement_weight
        + config.llm_weight
    )

    return (
        (
            engagement_score
            * config.engagement_weight
        )
        + (
            llm_score
            * config.llm_weight
        )
    ) / weight_total


def score_stage_b_response(
    response: StageBBatchResponse,
    engagement_scores: dict[str, float],
    config: StageBConfig,
) -> list[StageBScoredResult]:
    expected_post_ids = list(
        engagement_scores.keys()
    )

    validate_batch_response(
        response,
        expected_post_ids,
    )

    results: list[StageBScoredResult] = []

    for classification in response.classifications:
        engagement_score = engagement_scores[
            classification.post_id
        ]

        preliminary_score = (
            calculate_preliminary_score(
                engagement_score=engagement_score,
                relevance=classification.relevance,
                value=classification.value,
                config=config,
            )
        )

        results.append(
            StageBScoredResult(
                post_id=classification.post_id,
                engagement_score=engagement_score,
                relevance=classification.relevance,
                value=classification.value,
                kind=classification.kind,
                skip=classification.skip,
                skip_reason=classification.skip_reason,
                preliminary_score=preliminary_score,
            )
        )

    return results