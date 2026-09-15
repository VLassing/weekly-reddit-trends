import json
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from weekly_reddit_trends.models import Comment, Post
from weekly_reddit_trends.sources.base import RedditSource


class ArcticShiftSource(RedditSource):
    BASE_URL = "https://arctic-shift.photon-reddit.com"
    USER_AGENT = "weekly-reddit-trends/0.1 personal-read-only"
    TIMEOUT_SECONDS = 30
    MAX_RETRIES = 4

    @property
    def provider_name(self) -> str:
        return "arctic_shift"

    def fetch_posts(
        self,
        subreddit: str,
        start_at: datetime,
        end_at: datetime,
    ) -> list[Post]:
        start_at = self._as_utc(start_at)
        end_at = self._as_utc(end_at)

        if start_at >= end_at:
            raise ValueError("start_at must be before end_at")

        cursor = int(start_at.timestamp()) - 1
        end_epoch = int(end_at.timestamp())

        posts_by_id: dict[str, Post] = {}

        while cursor < end_epoch:
            payload = self._request_json(
                "/api/posts/search",
                {
                    "subreddit": subreddit,
                    "after": cursor,
                    "before": end_epoch,
                    "sort": "asc",
                    "limit": 100,
                },
            )

            raw_posts = payload.get("data") or []

            if not raw_posts:
                break

            fetched_at = datetime.now(timezone.utc)
            last_created = cursor

            for raw in raw_posts:
                post = self._normalize_post(raw, fetched_at)

                created_epoch = int(post.created_at.timestamp())
                last_created = max(last_created, created_epoch)

                if start_at <= post.created_at < end_at:
                    posts_by_id[post.id] = post

            if last_created <= cursor:
                cursor += 1
            else:
                cursor = last_created + 1

        return sorted(
            posts_by_id.values(),
            key=lambda post: post.created_at,
        )

    def fetch_posts_by_ids(
        self,
        post_ids: Sequence[str],
    ) -> list[Post]:
        clean_ids = [
            self._strip_prefix(post_id)
            for post_id in post_ids
        ]

        posts_by_id: dict[str, Post] = {}

        for offset in range(0, len(clean_ids), 500):
            chunk = clean_ids[offset : offset + 500]

            if not chunk:
                continue

            payload = self._request_json(
                "/api/posts/ids",
                {
                    "ids": ",".join(chunk),
                },
            )

            fetched_at = datetime.now(timezone.utc)

            for raw in payload.get("data") or []:
                post = self._normalize_post(raw, fetched_at)
                posts_by_id[post.id] = post

        return [
            posts_by_id[post_id]
            for post_id in clean_ids
            if post_id in posts_by_id
        ]

    def fetch_comments(
        self,
        post_id: str,
    ) -> list[Comment]:
        raise NotImplementedError(
            "Comment retrieval will be added after "
            "the post adapter is verified."
        )

    def _request_json(
        self,
        path: str,
        params: dict[str, object],
    ) -> dict:
        url = f"{self.BASE_URL}{path}?{urlencode(params)}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.USER_AGENT,
            },
        )

        for attempt in range(self.MAX_RETRIES):
            try:
                with urlopen(
                    request,
                    timeout=self.TIMEOUT_SECONDS,
                ) as response:
                    return json.loads(
                        response.read().decode("utf-8")
                    )

            except HTTPError as exc:
                retryable = (
                    exc.code == 429
                    or 500 <= exc.code < 600
                )

                if (
                    not retryable
                    or attempt == self.MAX_RETRIES - 1
                ):
                    raise

                delay = 2**attempt

                if exc.code == 429:
                    reset = exc.headers.get(
                        "X-RateLimit-Reset"
                    )

                    if reset:
                        try:
                            delay = max(
                                delay,
                                float(reset),
                            )
                        except ValueError:
                            pass

                time.sleep(
                    min(
                        max(delay, 1),
                        60,
                    )
                )

            except URLError:
                if attempt == self.MAX_RETRIES - 1:
                    raise

                time.sleep(2**attempt)

        raise RuntimeError(
            "Arctic Shift request failed unexpectedly"
        )

    def _normalize_post(
        self,
        raw: dict,
        fetched_at: datetime,
    ) -> Post:
        post_id = self._strip_prefix(
            str(raw["id"])
        )

        subreddit = str(
            raw.get("subreddit") or ""
        )

        title = str(
            raw.get("title") or ""
        )

        body = str(
            raw.get("selftext") or ""
        )

        score_raw = raw.get("score")
        score = (
            int(score_raw)
            if score_raw is not None
            else 0
        )

        comments_raw = raw.get("num_comments")

        if comments_raw is None:
            comments_raw = raw.get("comment_count")

        comment_count = (
            int(comments_raw)
            if comments_raw is not None
            else 0
        )

        created_at = self._parse_datetime(
            raw.get("created_utc")
        )

        domain_raw = raw.get("domain")
        domain = (
            str(domain_raw)
            if domain_raw
            else None
        )

        external_raw = raw.get("url")
        external_url = (
            str(external_raw)
            if external_raw
            else None
        )

        is_self_raw = raw.get("is_self")

        if is_self_raw is None:
            is_self = bool(
                domain
                and domain.lower().startswith("self.")
            )
        else:
            is_self = bool(is_self_raw)

        permalink_raw = raw.get("permalink")

        if permalink_raw:
            permalink = str(permalink_raw)

            if permalink.startswith("/"):
                permalink = (
                    "https://www.reddit.com"
                    + permalink
                )
        else:
            permalink = (
                "https://www.reddit.com/"
                f"r/{subreddit}/comments/{post_id}/"
            )

        crosspost_raw = (
            raw.get("crosspost_parent")
            or raw.get("crosspost_parent_id")
        )

        crosspost_parent = (
            self._strip_prefix(
                str(crosspost_raw)
            )
            if crosspost_raw
            else None
        )

        ratio_raw = raw.get("upvote_ratio")

        upvote_ratio = (
            float(ratio_raw)
            if ratio_raw is not None
            else None
        )

        flair_raw = raw.get("link_flair_text")
        flair = (
            str(flair_raw)
            if flair_raw
            else None
        )

        return Post(
            id=post_id,
            subreddit=subreddit,
            title=title,
            body=body,
            score=score,
            comment_count=comment_count,
            upvote_ratio=upvote_ratio,
            created_at=created_at,
            permalink=permalink,
            external_url=external_url,
            domain=domain,
            post_type=self._post_type(
                raw,
                is_self,
                external_url,
                domain,
            ),
            flair=flair,
            is_self=is_self,
            is_nsfw=bool(
                raw.get("over_18", False)
            ),
            is_stickied=bool(
                raw.get("stickied", False)
            ),
            is_locked=bool(
                raw.get("locked", False)
            ),
            is_crosspost=(
                crosspost_parent is not None
            ),
            crosspost_parent=crosspost_parent,
            removed_state=self._removed_state(
                raw,
                body,
                title,
            ),
            provider=self.provider_name,
            fetched_at=fetched_at,
            score_at_fetch=score,
        )

    @staticmethod
    def _post_type(
        raw: dict,
        is_self: bool,
        external_url: str | None,
        domain: str | None,
    ) -> str:
        if is_self:
            return "text"

        hint = str(
            raw.get("post_hint") or ""
        ).lower()

        url = (
            external_url or ""
        ).lower()

        domain_lower = (
            domain or ""
        ).lower()

        if (
            "image" in hint
            or domain_lower == "i.redd.it"
            or "/gallery/" in url
            or url.endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".webp",
                )
            )
        ):
            return "image"

        if (
            "video" in hint
            or domain_lower == "v.redd.it"
            or url.endswith(
                (
                    ".mp4",
                    ".webm",
                )
            )
        ):
            return "video"

        if external_url:
            return "link"

        return "other"

    @staticmethod
    def _removed_state(
        raw: dict,
        body: str,
        title: str,
    ) -> str:
        body_clean = body.strip().lower()
        title_clean = title.strip().lower()

        if (
            body_clean == "[deleted]"
            or title_clean == "[deleted]"
        ):
            return "deleted"

        if (
            body_clean == "[removed]"
            or title_clean == "[removed]"
            or raw.get("removed_by_category")
            is not None
        ):
            return "removed"

        return "none"

    @staticmethod
    def _strip_prefix(
        value: str,
    ) -> str:
        if value.startswith(
            (
                "t1_",
                "t3_",
            )
        ):
            return value[3:]

        return value

    @staticmethod
    def _as_utc(
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            raise ValueError(
                "datetime values must be timezone-aware"
            )

        return value.astimezone(
            timezone.utc
        )

    @staticmethod
    def _parse_datetime(
        value: object,
    ) -> datetime:
        if isinstance(
            value,
            (int, float),
        ):
            return datetime.fromtimestamp(
                value,
                tz=timezone.utc,
            )

        if isinstance(value, str):
            stripped = value.strip()

            try:
                return datetime.fromtimestamp(
                    float(stripped),
                    tz=timezone.utc,
                )
            except ValueError:
                pass

            return datetime.fromisoformat(
                stripped.replace(
                    "Z",
                    "+00:00",
                )
            ).astimezone(
                timezone.utc
            )

        raise ValueError(
            f"Unsupported created_utc value: {value!r}"
        )