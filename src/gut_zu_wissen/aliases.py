"""Short aliases for the offered facts (user decision 2026-11-20).

The dialog author sees `F1`, `F2` … instead of segment IDs, which are long and look alike
(revid + four digits) and were miscopied. Aliases follow the offer order (main article first,
then linked) and are unique per offer. The author's answer is mapped back to the segment IDs right
after parsing; everything downstream, the archive included, keeps the segment IDs. All functions
here are pure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

ALIAS_PREFIX = "F"

_ALIAS = re.compile(rf"{ALIAS_PREFIX}(\d+)", re.IGNORECASE)


def build_aliases(segment_ids: Iterable[str]) -> dict[str, str]:
    """Alias to segment ID in offer order, starting at 1. A repeated segment ID keeps its first
    alias."""
    aliases: dict[str, str] = {}
    for sid in segment_ids:
        if sid not in aliases.values():
            aliases[f"{ALIAS_PREFIX}{len(aliases) + 1}"] = sid
    return aliases


def normalize(token: str) -> str:
    """The author's token without surrounding spaces and brackets; alias-shaped tokens in canonical
    form (`[ f7 ]` becomes `F7`)."""
    bare = token.strip().removeprefix("[").removesuffix("]").strip()
    match = _ALIAS.fullmatch(bare)
    return f"{ALIAS_PREFIX}{int(match.group(1))}" if match else bare


def resolve(token: str, aliases: Mapping[str, str]) -> str:
    """Segment ID for an alias, or for an offered segment ID written out (unambiguous). Anything
    else comes back normalized, as the author wrote it, so the fact-ID gate reports it."""
    bare = normalize(token)
    if bare in aliases:
        return aliases[bare]
    return bare


def to_aliases(segment_ids: Sequence[str], aliases: Mapping[str, str]) -> list[str]:
    """Forward mapping for the author's view; IDs without an alias (not offered) are left out."""
    inverse = {sid: alias for alias, sid in aliases.items()}
    return [inverse[sid] for sid in segment_ids if sid in inverse]


def alias_range(aliases: Mapping[str, str]) -> str:
    """The aliases of the offer as a range, e.g. `F1–F29`."""
    names = list(aliases)
    if len(names) <= 1:
        return "".join(names)
    return f"{names[0]}–{names[-1]}"


def unknown_alias_problem(unknown: Sequence[str], aliases: Mapping[str, str]) -> str:
    """The author-facing text for identifiers that no offered fact has."""
    if len(unknown) == 1:
        head = f"Die Kennung {unknown[0]} gibt es im Angebot nicht."
    else:
        head = f"Die Kennungen {', '.join(unknown)} gibt es im Angebot nicht."
    return (
        f"{head} Verwende nur Kennungen aus dem Angebot ({alias_range(aliases)}); was du mit "
        "keiner angebotenen Kennung belegen kannst, lässt du weg."
    )


def replace_segment_ids(text: str, aliases: Mapping[str, str]) -> str:
    """Offered segment IDs inside a text (e.g. a judge's reason) replaced by their aliases."""
    for alias, sid in sorted(aliases.items(), key=lambda item: -len(item[1])):
        text = text.replace(sid, alias)
    return text
