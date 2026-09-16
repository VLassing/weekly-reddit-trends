from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .stage_b import (
    StageBBatchResponse,
    StageBInput,
    batch_stage_b_inputs,
)
from .stage_b_api import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    StageBApiUsage,
    classify_stage_b_batch,
)


@dataclass(frozen=True, slots=True)
class StageBRunResult:
    responses: tuple[StageBBatchResponse, ...]
    posts_processed: int
    batches_processed: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int
    truncated: bool


def run_stage_b_batches(
    inputs: list[StageBInput],
    *,
    batch_size: int,
    max_posts: int,
    max_batches: int,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> StageBRunResult:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if batch_size > 25:
        raise ValueError(
            "Stage B batch_size cannot exceed 25."
        )

    if max_posts <= 0:
        raise ValueError(
            "max_posts must be positive."
        )

    if max_batches <= 0:
        raise ValueError(
            "max_batches must be positive."
        )

    allowed_post_count = min(
        len(inputs),
        max_posts,
        batch_size * max_batches,
    )

    selected_inputs = inputs[
        :allowed_post_count
    ]

    batches = batch_stage_b_inputs(
        selected_inputs,
        batch_size,
    )

    responses: list[
        StageBBatchResponse
    ] = []

    total_usage = StageBApiUsage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        reasoning_tokens=0,
    )

    for batch in batches:
        api_result = classify_stage_b_batch(
            batch,
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
            max_attempts=max_attempts,
        )

        responses.append(
            api_result.response
        )

        total_usage = StageBApiUsage(
            input_tokens=(
                total_usage.input_tokens
                + api_result.usage.input_tokens
            ),
            output_tokens=(
                total_usage.output_tokens
                + api_result.usage.output_tokens
            ),
            total_tokens=(
                total_usage.total_tokens
                + api_result.usage.total_tokens
            ),
            reasoning_tokens=(
                total_usage.reasoning_tokens
                + api_result.usage.reasoning_tokens
            ),
        )

    return StageBRunResult(
        responses=tuple(responses),
        posts_processed=len(
            selected_inputs
        ),
        batches_processed=len(
            batches
        ),
        input_tokens=total_usage.input_tokens,
        output_tokens=total_usage.output_tokens,
        total_tokens=total_usage.total_tokens,
        reasoning_tokens=total_usage.reasoning_tokens,
        truncated=(
            len(selected_inputs)
            < len(inputs)
        ),
    )