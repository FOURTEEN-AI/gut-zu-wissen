"""Spike S1: all Jev question wording in one place (German and English variants).

The state is German either way. Rubric: docs/wikipedia-podcast-pipeline.md (Judge),
reaction criterion: docs/persona.md.
"""


def _paths(fact_ids: list[str]) -> str:
    return " und ".join(f"`fakten.{f}`" for f in fact_ids)


def _paths_en(fact_ids: list[str]) -> str:
    return " and ".join(f"`fakten.{f}`" for f in fact_ids)


def statement_question(lang: str, text: str, fact_ids: list[str]) -> dict:
    if lang == "de":
        p = _paths(fact_ids)
        return {
            "type": "noul",
            "instructions": f"Folgt die Aussage „{text}“ aus {p}?",
            "criteria": {
                "true": f"{p} sagt das aus oder es folgt direkt daraus. Andere Formulierungen "
                "und ausgeschriebene Zahlen ändern daran nichts.",
                "false": f"Die Aussage enthält etwas, das {p} nicht sagt, zum Beispiel eine "
                "abweichende Zahl, ein anderes Datum, einen anderen Namen oder eine zusätzliche "
                "Angabe. Was nur in `kontext` steht, gilt als nicht belegt.",
            },
        }
    p = _paths_en(fact_ids)
    return {
        "type": "noul",
        "instructions": f"Does the statement „{text}“ follow from {p}?",
        "criteria": {
            "true": f"{p} says this or it follows directly from it. Different wording and "
            "numbers written out as words do not change that.",
            "false": f"The statement contains something {p} does not say, for example a "
            "different number, a different date, a different name or an additional detail. "
            "Anything found only in `kontext` counts as not supported.",
        },
    }


REACTION_DE = (
    "eine Reaktion: ein Gefühl im Moment (Staunen, Neugier, Belustigung), eine Nachfrage, "
    "eine Bitte um Erklärung, eine Überleitung oder ein Wiederaufgreifen des eben Gesagten "
    "ohne neue Information"
)
REACTION_EN = (
    "a reaction: a feeling in the moment (amazement, curiosity, amusement), a follow-up "
    "question, a request for explanation, a transition, or picking up what was just said "
    "without new information"
)


def completeness_question(lang: str) -> dict:
    if lang == "de":
        return {
            "type": "noul",
            "instructions": "Ist jede Sachbehauptung und jede Prämisse einer Frage in "
            "`redebeitrag.text` unter `aussagen` erfasst?",
            "criteria": {
                "true": "Alles in `redebeitrag.text`, was nicht unter `aussagen` steht, ist "
                + REACTION_DE
                + ".",
                "false": "`redebeitrag.text` enthält etwas, das unter `aussagen` fehlt: eine "
                "neue Information, eine Wertung mit Faktengehalt oder eine Aussage der "
                "sprechenden Person über sich selbst, etwa über Erlebnisse, Wissen oder "
                "Vorlieben.",
            },
        }
    return {
        "type": "noul",
        "instructions": "Is every factual claim and every premise of a question in "
        "`redebeitrag.text` listed under `aussagen`?",
        "criteria": {
            "true": "Everything in `redebeitrag.text` that is not listed under `aussagen` is "
            + REACTION_EN
            + ".",
            "false": "`redebeitrag.text` contains something missing from `aussagen`: new "
            "information, an evaluation with factual content, or a statement by the speaker "
            "about themselves, such as experiences, knowledge or preferences.",
        },
    }


def d1_holistic(lang: str) -> dict:
    if lang == "de":
        return {
            "type": "noul",
            "instructions": "Vermittelt `nachher` dieselben Sachaussagen wie `vorher`?",
            "criteria": {
                "true": "Jede Sachaussage aus `vorher` steht auch in `nachher`, mit denselben "
                "Zahlen, Daten und Namen, und `nachher` bringt keine Sachaussage hinzu. "
                "Formulierung, Reihenfolge und Reaktionen spielen keine Rolle.",
                "false": "In `nachher` fehlt eine Sachaussage aus `vorher` oder ist verändert, "
                "oder `nachher` enthält eine Sachaussage, die `vorher` nicht enthält.",
            },
        }
    return {
        "type": "noul",
        "instructions": "Does `nachher` convey the same factual statements as `vorher`?",
        "criteria": {
            "true": "Every factual statement from `vorher` is also in `nachher`, with the same "
            "numbers, dates and names, and `nachher` adds no factual statement. Wording, order "
            "and reactions do not matter.",
            "false": "A factual statement from `vorher` is missing from `nachher` or changed, "
            "or `nachher` contains a factual statement that `vorher` does not.",
        },
    }


def d1_conveys(lang: str, statement: str) -> dict:
    if lang == "de":
        return {
            "type": "noul",
            "instructions": f"Vermittelt `nachher` die Aussage „{statement}“?",
            "criteria": {
                "true": "`nachher` sagt das, mit denselben Zahlen, Daten und Namen; die "
                "Formulierung darf abweichen.",
                "false": "`nachher` sagt das nicht oder mit einer anderen Zahl, einem anderen "
                "Datum oder einem anderen Namen.",
            },
        }
    return {
        "type": "noul",
        "instructions": f"Does `nachher` convey the statement „{statement}“?",
        "criteria": {
            "true": "`nachher` says this, with the same numbers, dates and names; the wording "
            "may differ.",
            "false": "`nachher` does not say this, or says it with a different number, date or "
            "name.",
        },
    }


def d1_nothing_added(lang: str) -> dict:
    if lang == "de":
        return {
            "type": "noul",
            "instructions": "Steht jede Sachaussage aus `nachher` auch in `vorher`?",
            "criteria": {
                "true": "Alles Sachliche in `nachher` kommt in `vorher` vor; Reaktionen und "
                "Überleitungen zählen nicht.",
                "false": "`nachher` enthält eine Sachaussage, die in `vorher` nicht vorkommt.",
            },
        }
    return {
        "type": "noul",
        "instructions": "Is every factual statement in `nachher` also in `vorher`?",
        "criteria": {
            "true": "Everything factual in `nachher` appears in `vorher`; reactions and "
            "transitions do not count.",
            "false": "`nachher` contains a factual statement that does not appear in `vorher`.",
        },
    }


# ---------------------------------------------------------------- round 2 (German only)


def sentence_is_reaction(sentence: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Ist der Satz „{sentence}“ aus `redebeitrag.text` eine reine Reaktion?",
        "criteria": {
            "true": f"Der Satz ist {REACTION_DE}.",
            "false": "Der Satz enthält eine Information, eine Wertung mit Faktengehalt oder "
            "eine Aussage der sprechenden Person über sich selbst, etwa über Erlebnisse, "
            "Wissen oder Vorlieben.",
        },
    }


def sentence_is_covered(sentence: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Gibt `aussagen` alles Sachliche im Satz „{sentence}“ wieder?",
        "criteria": {
            "true": "Jede Information und jede Prämisse einer Frage in diesem Satz steht in "
            "mindestens einer Aussage unter `aussagen`.",
            "false": "Der Satz enthält eine Information oder eine Prämisse, die in keiner "
            "Aussage unter `aussagen` steht.",
        },
    }


def d1_in_before(statement: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Vermittelt `vorher` die Aussage „{statement}“?",
        "criteria": {
            "true": "`vorher` sagt das, mit denselben Zahlen, Daten und Namen; die "
            "Formulierung darf abweichen.",
            "false": "`vorher` sagt das nicht oder mit einer anderen Zahl, einem anderen "
            "Datum oder einem anderen Namen.",
        },
    }
