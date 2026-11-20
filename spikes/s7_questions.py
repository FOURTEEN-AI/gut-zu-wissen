"""Spike S7, lever 5: Jev questions about the names, places, things and numbers of a statement.

Motivation (S1 follow-up, docs/spikes/s1-jev.md): the production statement question answered
0.98-0.99 for statements naming „Kometen“ or „Tagebau Goitzsche“, which are not in the cited fact
sentence. The production completeness question caught them only because it stayed uncertain.
These questions are meant as a safety net that would let a looser completeness threshold through.

Two families, both read with the same threshold t as the statement questions (true = fine):

- `entities`: one Noul per statement. Wording close to the user's proposal: does every name,
  place and number of the statement occur in the fact sentence itself? It asks about a set, which
  is one judgment by the human test, but it is the kind of "every X" question Jev is not built for
  (TypeSafe jaggedness page: counting and finding occurrences of a term), so `terms` exists too.
- `terms`: code does the exact lookup (s7_kosten.missing_terms: capitalised words and numbers of
  the statement, stem match against the fact sentence; the article title does not count, since the
  LLM judge never sees it). Only the terms code
  cannot find go to Jev, one Noul per term: is this term named in the fact sentence, possibly in
  another word form or with a word that means exactly the same thing? A statement whose terms code
  finds gets no `terms` question at all.

Both name the fact sentence by path (`fakten.<id>.satz`) and say that `kontext` does not count, as
the rubric of the LLM judge does. The state is the production state of `build_request`, unchanged.
"""


def _paths(fact_ids: list[str]) -> str:
    return " oder ".join(f"`fakten.{f}.satz`" for f in fact_ids)


def entity_question(text: str, fact_ids: list[str]) -> dict:
    paths = _paths(fact_ids)
    return {
        "type": "noul",
        "instructions": f"Kommt jeder Name, jeder Ort und jede Zahl der Aussage „{text}“ in "
        f"{paths} selbst vor?",
        "criteria": {
            "true": f"Jeder Name, jeder Ort und jede Zahl der Aussage steht in {paths}, auch in "
            "anderer Wortform oder als ausgeschriebene Zahl.",
            "false": "Die Aussage nennt einen Namen, einen Ort oder eine Zahl, die in "
            f"{paths} nicht steht, etwa weil sie nur in `kontext` oder gar nicht vorkommt.",
        },
    }


def term_question(term: str, text: str, fact_ids: list[str]) -> dict:
    paths = _paths(fact_ids)
    return {
        "type": "noul",
        "instructions": f"Nennt {paths} „{term}“ aus der Aussage „{text}“?",
        "criteria": {
            "true": f"{paths} nennt „{term}“ selbst, auch in anderer Wortform (etwa Plural oder "
            "Genitiv) oder mit einem Wort, das genau dasselbe bezeichnet.",
            "false": f"„{term}“ steht nicht in {paths}: dort steht nur ein Oberbegriff, ein "
            f"verwandter Begriff oder gar nichts dazu, oder „{term}“ steht nur in `kontext`.",
        },
    }
