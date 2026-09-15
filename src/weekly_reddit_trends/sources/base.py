from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from weekly_reddit_trends.models import Comment, Post


class RedditSource(ABC):
    """Provider-agnostic interface for normalized Reddit data."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable name stored in normalized provenance fields."""
        raise NotImplementedError

    @abstractmethod
    def fetch_posts(
        self,
        subreddit: str,
        start_at: datetime,
        end_at: datetime,
    ) -> list[Post]:
        """Fetch all available posts created inside the requested window."""
        raise NotImplementedError

    @abstractmethod
    def fetch_posts_by_ids(
        self,
        post_ids: Sequence[str],
    ) -> list[Post]:
        """Fetch or hydrate specific posts by Reddit post ID."""
        raise NotImplementedError

    @abstractmethod
    def fetch_comments(
        self,
        post_id: str,
    ) -> list[Comment]:
        """Fetch the available normalized comment tree for one post."""
        raise NotImplementedError
