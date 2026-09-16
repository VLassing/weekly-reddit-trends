from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)


TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref_src",
    "ref_url",
    "source",
}


def normalize_external_url(
    value: str | None,
) -> str | None:
    """
    Produce a conservative canonical form for external URLs.

    Normalization intentionally avoids aggressive semantic rewrites.
    It only removes common tracking differences and obvious
    representation differences.

    HTTP and HTTPS are treated as equivalent for deduplication.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        parsed = urlsplit(value)
    except ValueError:
        return None

    if parsed.scheme.lower() not in {
        "http",
        "https",
    }:
        return None

    hostname = (
        parsed.hostname or ""
    ).lower()

    if not hostname:
        return None

    if hostname.startswith("www."):
        hostname = hostname[4:]

    port = parsed.port

    if port is not None:
        default_port = (
            parsed.scheme.lower() == "http"
            and port == 80
        ) or (
            parsed.scheme.lower() == "https"
            and port == 443
        )

        if not default_port:
            hostname = (
                f"{hostname}:{port}"
            )

    path = parsed.path or "/"

    while (
        len(path) > 1
        and path.endswith("/")
    ):
        path = path[:-1]

    clean_query = []

    for key, val in parse_qsl(
        parsed.query,
        keep_blank_values=True,
    ):
        key_lower = key.lower()

        if key_lower.startswith("utm_"):
            continue

        if key_lower in TRACKING_QUERY_KEYS:
            continue

        clean_query.append(
            (key, val)
        )

    clean_query.sort(
        key=lambda pair: (
            pair[0].lower(),
            pair[1],
        )
    )

    query = urlencode(
        clean_query,
        doseq=True,
    )

    # Deliberately omit scheme so http:// and https://
    # versions of the same resource deduplicate.
    return urlunsplit(
        (
            "",
            hostname,
            path,
            query,
            "",
        )
    ).lstrip("//")


class _UnionFind:
    def __init__(
        self,
        size: int,
    ) -> None:
        self.parent = list(
            range(size)
        )

        self.rank = [
            0
            for _ in range(size)
        ]

    def find(
        self,
        value: int,
    ) -> int:
        if self.parent[value] != value:
            self.parent[value] = (
                self.find(
                    self.parent[value]
                )
            )

        return self.parent[value]

    def union(
        self,
        left: int,
        right: int,
    ) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if (
            self.rank[left_root]
            < self.rank[right_root]
        ):
            left_root, right_root = (
                right_root,
                left_root,
            )

        self.parent[right_root] = (
            left_root
        )

        if (
            self.rank[left_root]
            == self.rank[right_root]
        ):
            self.rank[left_root] += 1


def _item_post_id(
    item: dict,
) -> str:
    stage_b = item.get("stage_b")
    post = item.get("post")

    if not isinstance(
        stage_b,
        dict,
    ):
        raise ValueError(
            "Dedup item is missing "
            "stage_b data."
        )

    if not isinstance(
        post,
        dict,
    ):
        raise ValueError(
            "Dedup item is missing "
            "hydrated post data."
        )

    stage_b_id = str(
        stage_b.get("post_id") or ""
    ).strip()

    post_id = str(
        post.get("id") or ""
    ).strip()

    if not stage_b_id:
        raise ValueError(
            "Stage B item has no post_id."
        )

    if not post_id:
        raise ValueError(
            "Hydrated post has no id."
        )

    if stage_b_id != post_id:
        raise ValueError(
            "Stage B post_id does not match "
            "hydrated post id: "
            f"{stage_b_id!r} != {post_id!r}"
        )

    return post_id


def _crosspost_parent(
    item: dict,
) -> str | None:
    raw = (
        item["post"].get(
            "crosspost_parent"
        )
    )

    if raw is None:
        return None

    value = str(raw).strip()

    if not value:
        return None

    if value.startswith(
        (
            "t1_",
            "t3_",
        )
    ):
        value = value[3:]

    return value


def _normalized_url(
    item: dict,
) -> str | None:
    return normalize_external_url(
        item["post"].get(
            "external_url"
        )
    )


def _representative_sort_key(
    item: dict,
) -> tuple:
    stage_b = item["stage_b"]

    return (
        -float(
            stage_b.get(
                "preliminary_score",
                0.0,
            )
        ),
        -int(
            stage_b.get(
                "relevance",
                0,
            )
        ),
        -int(
            stage_b.get(
                "value",
                0,
            )
        ),
        -float(
            stage_b.get(
                "engagement_score",
                0.0,
            )
        ),
        str(
            stage_b.get(
                "subreddit",
                "",
            )
        ).casefold(),
        str(
            stage_b.get(
                "post_id",
                "",
            )
        ),
    )


def _group_id(
    post_ids: Sequence[str],
) -> str:
    payload = "\n".join(
        sorted(post_ids)
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).hexdigest()[:12]

    return f"dedup-{digest}"


def build_dedup_groups(
    items: Sequence[dict],
) -> list[dict]:
    """
    Build connected deterministic duplicate groups.

    Relationships currently used:

    1. Same post ID.
    2. Same conservatively normalized external URL.
    3. Crosspost points at another shortlisted post.
    4. Multiple shortlist posts share the same
       crosspost parent.

    Connected relationships are merged transitively.
    No post is discarded.
    """

    items = list(items)

    if not items:
        return []

    union_find = _UnionFind(
        len(items)
    )

    post_ids = [
        _item_post_id(item)
        for item in items
    ]

    post_id_to_indexes: dict[
        str,
        list[int],
    ] = defaultdict(list)

    url_to_indexes: dict[
        str,
        list[int],
    ] = defaultdict(list)

    parent_to_indexes: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for index, item in enumerate(
        items
    ):
        post_id_to_indexes[
            post_ids[index]
        ].append(index)

        normalized_url = (
            _normalized_url(item)
        )

        if normalized_url:
            url_to_indexes[
                normalized_url
            ].append(index)

        parent = _crosspost_parent(
            item
        )

        if parent:
            parent_to_indexes[
                parent
            ].append(index)

    # Same post ID.
    for indexes in (
        post_id_to_indexes.values()
    ):
        first = indexes[0]

        for other in indexes[1:]:
            union_find.union(
                first,
                other,
            )

    # Same normalized external URL.
    for indexes in (
        url_to_indexes.values()
    ):
        if len(indexes) < 2:
            continue

        first = indexes[0]

        for other in indexes[1:]:
            union_find.union(
                first,
                other,
            )

    # Multiple crossposts sharing the same parent.
    for indexes in (
        parent_to_indexes.values()
    ):
        if len(indexes) < 2:
            continue

        first = indexes[0]

        for other in indexes[1:]:
            union_find.union(
                first,
                other,
            )

    # Crosspost pointing directly at another post
    # that exists in the shortlist.
    for parent_id, indexes in (
        parent_to_indexes.items()
    ):
        parent_indexes = (
            post_id_to_indexes.get(
                parent_id
            )
        )

        if not parent_indexes:
            continue

        parent_index = (
            parent_indexes[0]
        )

        for child_index in indexes:
            union_find.union(
                parent_index,
                child_index,
            )

    component_indexes: dict[
        int,
        list[int],
    ] = defaultdict(list)

    for index in range(
        len(items)
    ):
        root = union_find.find(
            index
        )

        component_indexes[
            root
        ].append(index)

    groups = []

    for indexes in (
        component_indexes.values()
    ):
        group_items = [
            items[index]
            for index in indexes
        ]

        ordered_items = sorted(
            group_items,
            key=_representative_sort_key,
        )

        representative = (
            ordered_items[0]
        )

        member_ids = [
            _item_post_id(item)
            for item in ordered_items
        ]

        shared_urls = []

        group_url_members: dict[
            str,
            list[str],
        ] = defaultdict(list)

        for item in ordered_items:
            normalized_url = (
                _normalized_url(item)
            )

            if normalized_url:
                group_url_members[
                    normalized_url
                ].append(
                    _item_post_id(
                        item
                    )
                )

        for url, ids in sorted(
            group_url_members.items()
        ):
            if len(ids) < 2:
                continue

            shared_urls.append(
                {
                    "normalized_url": url,
                    "post_ids": sorted(
                        ids
                    ),
                }
            )

        crosspost_links = []

        for item in ordered_items:
            parent = (
                _crosspost_parent(
                    item
                )
            )

            if not parent:
                continue

            crosspost_links.append(
                {
                    "post_id": (
                        _item_post_id(
                            item
                        )
                    ),
                    "parent_id": parent,
                }
            )

        crosspost_links.sort(
            key=lambda link: (
                link["parent_id"],
                link["post_id"],
            )
        )

        parent_counts = Counter(
            link["parent_id"]
            for link in crosspost_links
        )

        shared_crosspost_parents = [
            {
                "parent_id": parent_id,
                "post_ids": sorted(
                    link["post_id"]
                    for link
                    in crosspost_links
                    if (
                        link[
                            "parent_id"
                        ]
                        == parent_id
                    )
                ),
            }
            for parent_id, count
            in sorted(
                parent_counts.items()
            )
            if count >= 2
        ]

        duplicate_signals = []

        if len(set(member_ids)) < len(
            member_ids
        ):
            duplicate_signals.append(
                "same_post_id"
            )

        if shared_urls:
            duplicate_signals.append(
                "same_external_url"
            )

        group_member_set = set(
            member_ids
        )

        if any(
            link["parent_id"]
            in group_member_set
            for link in crosspost_links
        ):
            duplicate_signals.append(
                "crosspost_parent_in_group"
            )

        if shared_crosspost_parents:
            duplicate_signals.append(
                "shared_crosspost_parent"
            )

        representative_id = (
            _item_post_id(
                representative
            )
        )

        groups.append(
            {
                "group_id": _group_id(
                    member_ids
                ),
                "representative_post_id": (
                    representative_id
                ),
                "member_post_ids": (
                    member_ids
                ),
                "size": len(
                    member_ids
                ),
                "is_duplicate_group": (
                    len(member_ids) > 1
                ),
                "duplicate_signals": (
                    duplicate_signals
                ),
                "shared_external_urls": (
                    shared_urls
                ),
                "crosspost_links": (
                    crosspost_links
                ),
                "shared_crosspost_parents": (
                    shared_crosspost_parents
                ),
                "representative": (
                    representative
                ),
                "members": ordered_items,
            }
        )

    groups.sort(
        key=lambda group: (
            -float(
                group[
                    "representative"
                ][
                    "stage_b"
                ].get(
                    "preliminary_score",
                    0.0,
                )
            ),
            group[
                "representative_post_id"
            ],
        )
    )

    return groups


def summarize_dedup_groups(
    groups: Sequence[dict],
) -> dict:
    groups = list(groups)

    input_posts = sum(
        group["size"]
        for group in groups
    )

    duplicate_groups = [
        group
        for group in groups
        if group[
            "is_duplicate_group"
        ]
    ]

    duplicate_posts = sum(
        group["size"] - 1
        for group in duplicate_groups
    )

    return {
        "input_posts": input_posts,
        "groups": len(groups),
        "standalone_groups": (
            len(groups)
            - len(duplicate_groups)
        ),
        "duplicate_groups": len(
            duplicate_groups
        ),
        "posts_merged_as_duplicates": (
            duplicate_posts
        ),
        "largest_group_size": max(
            (
                group["size"]
                for group in groups
            ),
            default=0,
        ),
    }