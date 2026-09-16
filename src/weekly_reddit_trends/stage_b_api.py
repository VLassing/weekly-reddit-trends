from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .stage_b import (
    STAGE_B_INSTRUCTIONS,
    StageBBatchResponse,
    StageBInput,
    serialize_stage_b_batch,
    validate_batch_response,
)


DEFAULT_MODEL = "gpt-5.6-luna"

HARD_MAX_BATCH_SIZE = 25
DEFAULT_MAX_OUTPUT_TOKENS = 3000
DEFAULT_MAX_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class StageBApiUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True, slots=True)
class StageBApiResult:
    response: StageBBatchResponse
    usage: StageBApiUsage
    attempts: int


def create_stage_b_client() -> OpenAI:
    return OpenAI(
        max_retries=0,
    )


def _validate_api_settings(
    batch: list[StageBInput],
    max_output_tokens: int,
    max_attempts: int,
) -> None:
    if not batch:
        raise ValueError(
            "Stage B API batch cannot be empty."
        )

    if len(batch) > HARD_MAX_BATCH_SIZE:
        raise ValueError(
            f"Stage B API batch contains {len(batch)} posts. "
            f"Hard limit is {HARD_MAX_BATCH_SIZE}."
        )

    if max_output_tokens < 16:
        raise ValueError(
            "max_output_tokens must be at least 16."
        )

    if max_output_tokens > 5000:
        raise ValueError(
            "Refusing Stage B API call with "
            "max_output_tokens above 5000."
        )

    if max_attempts < 1:
        raise ValueError(
            "max_attempts must be at least 1."
        )

    if max_attempts > 2:
        raise ValueError(
            "Stage B API allows at most 2 attempts "
            "per batch."
        )


def _usage_from_response(
    response,
) -> StageBApiUsage:
    usage = response.usage

    if usage is None:
        return StageBApiUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            reasoning_tokens=0,
        )

    reasoning_tokens = 0

    if usage.output_tokens_details is not None:
        reasoning_tokens = (
            usage
            .output_tokens_details
            .reasoning_tokens
            or 0
        )

    return StageBApiUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _repair_unambiguous_post_id_mismatch(
    response: StageBBatchResponse,
    expected_post_ids: list[str],
) -> StageBBatchResponse:
    """
    Repair one clearly unambiguous model typo in a post ID.

    Repair is allowed only when:
    - the response contains the expected number of classifications
    - expected IDs are unique
    - exactly one expected ID is missing
    - exactly one unexpected ID is present
    - the unexpected ID occurs exactly once

    Example:

        expected:
        1wbwvfw

        returned:
        1wbvfwv

    If more than one ID differs, no repair is attempted.
    Normal validation will then fail the response.
    """

    if len(response.classifications) != len(
        expected_post_ids
    ):
        return response

    expected_set = set(expected_post_ids)

    if len(expected_set) != len(
        expected_post_ids
    ):
        return response

    returned_ids = [
        classification.post_id
        for classification
        in response.classifications
    ]

    returned_set = set(returned_ids)

    missing_ids = (
        expected_set
        - returned_set
    )

    unexpected_ids = (
        returned_set
        - expected_set
    )

    if (
        len(missing_ids) != 1
        or len(unexpected_ids) != 1
    ):
        return response

    missing_id = next(iter(missing_ids))
    unexpected_id = next(
        iter(unexpected_ids)
    )

    if returned_ids.count(
        unexpected_id
    ) != 1:
        return response

    repaired_classifications = []

    for classification in (
        response.classifications
    ):
        if (
            classification.post_id
            == unexpected_id
        ):
            repaired_classifications.append(
                classification.model_copy(
                    update={
                        "post_id": missing_id,
                    }
                )
            )
        else:
            repaired_classifications.append(
                classification
            )

    return response.model_copy(
        update={
            "classifications": (
                repaired_classifications
            ),
        }
    )


def classify_stage_b_batch(
    batch: list[StageBInput],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> StageBApiResult:
    _validate_api_settings(
        batch=batch,
        max_output_tokens=max_output_tokens,
        max_attempts=max_attempts,
    )

    api_client = (
        client
        if client is not None
        else create_stage_b_client()
    )

    expected_post_ids = [
        item.post_id
        for item in batch
    ]

    payload = serialize_stage_b_batch(
        batch
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            raw_response = (
                api_client.responses.parse(
                    model=model,
                    instructions=(
                        STAGE_B_INSTRUCTIONS
                    ),
                    input=payload,
                    text_format=(
                        StageBBatchResponse
                    ),
                    reasoning={
                        "effort": "none",
                    },
                    max_output_tokens=(
                        max_output_tokens
                    ),
                    store=False,
                )
            )

            parsed = raw_response.output_parsed

            if parsed is None:
                raise ValueError(
                    "OpenAI returned no parsed "
                    "Stage B Structured Output."
                )

            parsed = (
                _repair_unambiguous_post_id_mismatch(
                    parsed,
                    expected_post_ids,
                )
            )

            validate_batch_response(
                parsed,
                expected_post_ids,
            )

            return StageBApiResult(
                response=parsed,
                usage=_usage_from_response(
                    raw_response
                ),
                attempts=attempt,
            )

        except Exception as exc:
            last_error = exc

            if attempt >= max_attempts:
                break

    if last_error is None:
        raise RuntimeError(
            "Stage B API failed without an error."
        )

    raise RuntimeError(
        f"Stage B API failed after "
        f"{max_attempts} attempt(s): "
        f"{type(last_error).__name__}: "
        f"{last_error}"
    ) from last_error