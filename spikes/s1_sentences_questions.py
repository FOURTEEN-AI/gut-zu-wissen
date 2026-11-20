"""Per-sentence completeness, follow-up to spike S1: all wording variants in one place.

User decision 2026-11-20: test whether asking Jev about each sentence of `redebeitrag.text`
lets it decide more turns alone than the one completeness question per turn. Every question has
true = the sentence is fine (a pure reaction, or covered by `aussagen`), so code reads all of them
with the same threshold t as the statement questions.

The reaction criterion comes from docs/persona.md ("Reaktionskriterium (belegfrei)"); the
wording of the allowed and not-allowed lists is the one of spikes/s1_completeness_questions.py.
The sentence is named in the question itself (one question per sentence); the state is the
production state of gut_zu_wissen.judge.build_request, unchanged.

Variants (code combines the answers; `sentence_p` in s1_sentences.py):
- `either`: one Noul per sentence, "reine Reaktion oder vollständig erfasst";
- `split`: two Nouls per sentence, `reaction` and `covered`; sentence p = max of the two;
- `split_ctx`: as `split`, with the sentence before and after in structured instructions.
"""

ALLOWED = (
    "ein Gefühl im Moment (Staunen, Neugier, Belustigung), eine Nachfrage, eine Bitte um "
    "Erklärung, eine Überleitung oder ein Wiederaufgreifen des eben Gesagten ohne neue "
    "Information, etwa als Ausruf oder Rückfrage"
)
NOT_ALLOWED = (
    "eine neue Information, etwa eine Zahl, ein Name, ein Ereignis oder eine Eigenschaft; eine "
    "Zusammenfassung des Gesagten als eigene Feststellung; eine Wertung des Inhalts mit "
    "Faktengehalt, etwa „Das ist ja das größte …“; eine Aussage der sprechenden Person über "
    "sich selbst über den Moment hinaus, etwa „Das kenne ich“ oder „Davon habe ich gehört“"
)
PREMISE = (
    "Bei einer Frage zählt ihre Prämisse als Inhalt: „Warum leuchtet der Stein im Dunkeln?“ "
    "teilt mit, dass der Stein im Dunkeln leuchtet."
)


def either(sentence: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Ist der Satz „{sentence}“ aus `redebeitrag.text` eine reine Reaktion "
        "oder vollständig unter `aussagen` erfasst?",
        "criteria": {
            "true": "Der Satz ist eine reine Reaktion ohne eigenen Inhalt: "
            + ALLOWED
            + ". Oder alles, was der Satz an Inhalt mitteilt, steht in einer Aussage unter "
            "`aussagen`; andere Formulierungen genügen. " + PREMISE,
            "false": "Der Satz teilt etwas mit, das in keiner Aussage unter `aussagen` steht: "
            + NOT_ALLOWED
            + "; oder die Prämisse einer Frage fehlt unter `aussagen`.",
        },
    }


def reaction(sentence: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Ist der Satz „{sentence}“ aus `redebeitrag.text` eine reine Reaktion "
        "ohne eigenen Inhalt?",
        "criteria": {
            "true": "Der Satz ist " + ALLOWED + ".",
            "false": "Der Satz teilt etwas mit: "
            + NOT_ALLOWED
            + "; oder er ist eine Frage mit einer Sachangabe als Prämisse. "
            + PREMISE,
        },
    }


def covered(sentence: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Steht alles, was der Satz „{sentence}“ aus `redebeitrag.text` an "
        "Inhalt mitteilt, in einer Aussage unter `aussagen`?",
        "criteria": {
            "true": "Jede Sachangabe des Satzes steht in einer Aussage unter `aussagen`; andere "
            "Formulierungen genügen. " + PREMISE,
            "false": "Der Satz enthält eine Sachangabe, eine Wertung mit Faktengehalt oder eine "
            "Prämisse, die in keiner Aussage unter `aussagen` steht.",
        },
    }


def _with_context(question: dict, before: str | None, after: str | None) -> dict:
    instructions: dict = {"frage": question["instructions"]}
    instructions["satz_davor"] = before if before else "(Der Satz steht am Anfang.)"
    instructions["satz_danach"] = after if after else "(Der Satz steht am Ende.)"
    return {**question, "instructions": instructions}


def reaction_ctx(sentence: str, before: str | None, after: str | None) -> dict:
    return _with_context(reaction(sentence), before, after)


def covered_ctx(sentence: str, before: str | None, after: str | None) -> dict:
    return _with_context(covered(sentence), before, after)


def sentence_questions(variant: str, sentences: list[str]) -> dict[str, dict]:
    """Questions of one variant for all sentences of a turn; keys `{variant}_{kind}_{n}`."""
    out: dict[str, dict] = {}
    for n, s in enumerate(sentences):
        before = sentences[n - 1] if n else None
        after = sentences[n + 1] if n + 1 < len(sentences) else None
        if variant == "either":
            out[f"either_ok_{n}"] = either(s)
        elif variant == "split":
            out[f"split_reaction_{n}"] = reaction(s)
            out[f"split_covered_{n}"] = covered(s)
        elif variant == "split_ctx":
            out[f"split_ctx_reaction_{n}"] = reaction_ctx(s, before, after)
            out[f"split_ctx_covered_{n}"] = covered_ctx(s, before, after)
        else:
            raise ValueError(variant)
    return out


VARIANTS = ("either", "split", "split_ctx")
