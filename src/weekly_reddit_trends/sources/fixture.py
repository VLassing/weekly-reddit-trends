from collections.abc import Sequence
from datetime import datetime

from weekly_reddit_trends.models import Comment, Post
from weekly_reddit_trends.sources.base import RedditSource


class FixtureSource(RedditSource):
    """In-memory Reddit source for deterministic tests."""

    def __init__(
        self,
        posts: Sequence[Post] = (),
        comments: Sequence[Comment] = (),
    ) -> None:
        self._posts = list(posts)
        self._comments = list(comments)

    @property
    def provider_name(self) -> str:
        return "fixture"

    def fetch_posts(
        self,
        subreddit: str,
        start_at: datetime,
        end_at: datetime,
    ) -> list[Post]:
        return [
            post
            for post in self._posts
            if post.subreddit.casefold() == subreddit.casefold()
            and start_at <= post.created_at < end_at
        ]

    def fetch_posts_by_ids(
        self,
        post_ids: Sequence[str],
    ) -> list[Post]:
        posts_by_id = {post.id: post for post in self._posts}
        return [
            posts_by_id[post_id]
            for post_id in post_ids
            if post_id in posts_by_id
        ]

    def fetch_comments(
        self,
        post_id: str,
    ) -> list[Comment]:
        return [
            comment
            for comment in self._comments
            if comment.post_id == post_id
        ]
