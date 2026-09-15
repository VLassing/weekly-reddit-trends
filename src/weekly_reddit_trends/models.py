from dataclasses import dataclass
from datetime import datetime
from typing import Literal


PostType = Literal["text", "link", "image", "video", "other"]
RemovedState = Literal["none", "removed", "deleted"]


@dataclass(slots=True)
class Post:
    id: str
    subreddit: str
    title: str
    body: str
    score: int
    comment_count: int
    upvote_ratio: float | None
    created_at: datetime
    permalink: str
    external_url: str | None
    domain: str | None
    post_type: PostType
    flair: str | None
    is_self: bool
    is_nsfw: bool
    is_stickied: bool
    is_locked: bool
    is_crosspost: bool
    crosspost_parent: str | None
    removed_state: RemovedState
    provider: str
    fetched_at: datetime
    score_at_fetch: int


@dataclass(slots=True)
class Comment:
    id: str
    post_id: str
    parent_id: str | None
    body: str
    score: int
    created_at: datetime
    depth: int
    is_submitter: bool
    removed_state: RemovedState
    reply_count: int
    provider: str
    fetched_at: datetime
