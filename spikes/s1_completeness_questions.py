"""Completeness question follow-up to spike S1: all wording variants in one place.

`old` is the production question before the rework (src/gut_zu_wissen/judge.py, from S1).
The reaction criterion comes from docs/persona.md ("Reaktionskriterium (belegfrei)", "Bewusst
ausgeschlossen", Felix: "fasst zusammen nur mit Fakt-IDs"). Every variant has true = complete, so
code can read it with the same threshold as the statement questions, except `adds`, whose
true = something added (code reads 1 - p).
"""

ALLOWED = (
    "ein Gefühl im Moment (Staunen, Neugier, Belustigung), eine Nachfrage, eine Bitte um "
    "Erklärung, eine Überleitung oder ein Wiederaufgreifen des eben Gesagten ohne neue "
    "Information, etwa als Ausruf oder Rückfrage"
)
NOT_ALLOWED = (
    "eine neue Information, etwa eine Zahl, ein Name, ein Ereignis oder eine Eigenschaft; eine "
    "Zusammenfassung des Gesagten als eigene Feststellung; eine Wertung des Inhalts mit "
    "Faktengehalt, etwa „Das ist ja das größte …“; eine Ankündigung mit Inhalt; eine Aussage der "
    "sprechenden Person über sich selbst über den Moment hinaus, etwa „Das kenne ich“ oder "
    "„Davon habe ich gehört“"
)

OLD_REACTION = (
    "eine Reaktion: ein Gefühl im Moment (Staunen, Neugier, Belustigung), eine Nachfrage, "
    "eine Bitte um Erklärung, eine Überleitung oder ein Wiederaufgreifen des eben Gesagten "
    "ohne neue Information"
)


def old() -> dict:
    return {
        "type": "noul",
        "instructions": "Ist jede Sachbehauptung und jede Prämisse einer Frage in "
        "`redebeitrag.text` unter `aussagen` erfasst?",
        "criteria": {
            "true": "Alles in `redebeitrag.text`, was nicht unter `aussagen` steht, ist "
            + OLD_REACTION
            + ".",
            "false": "`redebeitrag.text` enthält etwas, das unter `aussagen` fehlt: eine neue "
            "Information, eine Wertung mit Faktengehalt oder eine Aussage der sprechenden "
            "Person über sich selbst, etwa über Erlebnisse, Wissen oder Vorlieben.",
        },
    }


def content() -> dict:
    """Asks about content only; says plainly that reactions are not content."""
    return {
        "type": "noul",
        "instructions": "Steht alles, was `redebeitrag.text` an Inhalt mitteilt, auch unter "
        "`aussagen`?",
        "criteria": {
            "true": "Jede Sachangabe in `redebeitrag.text` steht unter `aussagen`; andere "
            "Formulierungen genügen. Bei einer Frage zählt nur ihre Prämisse als Inhalt. Sätze "
            "ohne Inhalt dürfen unter `aussagen` fehlen: " + ALLOWED + ".",
            "false": "`redebeitrag.text` teilt etwas mit, das unter `aussagen` fehlt: "
            + NOT_ALLOWED
            + ".",
        },
    }


def structured() -> dict:
    """Same judgement as `content`, with labelled lists instead of prose."""
    return {
        "type": "noul",
        "instructions": {
            "frage": "Steht alles, was `redebeitrag.text` an Inhalt mitteilt, auch unter "
            "`aussagen`?",
            "zaehlt_nicht_als_inhalt": [
                "Gefühl im Moment (Staunen, Neugier, Belustigung)",
                "Nachfrage",
                "Bitte um Erklärung",
                "Überleitung",
                "Wiederaufgreifen des eben Gesagten ohne neue Information, etwa als Ausruf "
                "oder Rückfrage",
            ],
            "zaehlt_als_inhalt": [
                "neue Information, etwa eine Zahl, ein Name, ein Ereignis oder eine Eigenschaft",
                "Prämisse einer Frage",
                "Zusammenfassung des Gesagten als eigene Feststellung",
                "Wertung des Inhalts mit Faktengehalt, etwa „Das ist ja das größte …“",
                "Ankündigung mit Inhalt",
                "Aussage der sprechenden Person über sich selbst über den Moment hinaus, etwa "
                "„Das kenne ich“",
            ],
        },
        "criteria": {
            "true": "Alles, was als Inhalt zählt, steht unter `aussagen`; andere Formulierungen "
            "genügen.",
            "false": "Etwas, das als Inhalt zählt, fehlt unter `aussagen`.",
        },
    }


def adds() -> dict:
    """Reversed polarity: true = the turn adds content (code reads 1 - p)."""
    return {
        "type": "noul",
        "instructions": "Teilt `redebeitrag.text` etwas mit, das unter `aussagen` fehlt?",
        "criteria": {
            "true": "`redebeitrag.text` enthält mindestens eines davon, und es steht nicht "
            "unter `aussagen`: " + NOT_ALLOWED + "; oder die Prämisse einer Frage.",
            "false": "Jede Sachangabe steht unter `aussagen`, auch in anderen Worten. Alles "
            "Übrige ist " + ALLOWED + ".",
        },
    }


# ---------------------------------------------------------------- round 2: against `fakten`
# Round 1 showed that the run's labels judge the text against the fact sentences, while the
# questions above judge it against the author's statement list. Round 2 asks the rubric directly.

BEYOND_FACTS = (
    "eine Information, die `fakten` nicht enthält, etwa eine Zahl, ein Name, ein Ereignis oder "
    "eine Eigenschaft; eine Wertung des Inhalts mit Faktengehalt, etwa „Das ist ja das größte "
    "…“; eine Aussage der sprechenden Person über sich selbst über den Moment hinaus, etwa „Das "
    "kenne ich“ oder „Davon habe ich gehört“"
)


def facts() -> dict:
    return {
        "type": "noul",
        "instructions": "Teilt `redebeitrag.text` nur Inhalte mit, die in `fakten` stehen?",
        "criteria": {
            "true": "Jede Sachangabe in `redebeitrag.text` und jede Prämisse einer Frage steht "
            "in `fakten` oder folgt direkt daraus; andere Formulierungen und ausgeschriebene "
            "Zahlen genügen. Sätze ohne Inhalt brauchen keinen Beleg: " + ALLOWED + ".",
            "false": "`redebeitrag.text` teilt etwas mit, das nicht in `fakten` steht: "
            + BEYOND_FACTS
            + ". Was nur in `kontext` steht, gilt als nicht belegt.",
        },
    }


def facts_structured() -> dict:
    return {
        "type": "noul",
        "instructions": {
            "frage": "Teilt `redebeitrag.text` nur Inhalte mit, die in `fakten` stehen?",
            "braucht_keinen_beleg": [
                "Gefühl im Moment (Staunen, Neugier, Belustigung)",
                "Nachfrage",
                "Bitte um Erklärung",
                "Überleitung",
                "Wiederaufgreifen des eben Gesagten ohne neue Information, etwa als Ausruf "
                "oder Rückfrage",
            ],
            "braucht_beleg_in_fakten": [
                "Sachangabe, etwa eine Zahl, ein Name, ein Ereignis oder eine Eigenschaft",
                "Prämisse einer Frage",
                "Wertung des Inhalts mit Faktengehalt, etwa „Das ist ja das größte …“",
                "Ankündigung mit Inhalt",
            ],
            "nie_erlaubt": "Aussage der sprechenden Person über sich selbst über den Moment "
            "hinaus, etwa „Das kenne ich“",
            "kein_beleg": "Was nur in `kontext` steht.",
        },
        "criteria": {
            "true": "Alles, was einen Beleg braucht, steht in `fakten` oder folgt direkt "
            "daraus; andere Formulierungen und ausgeschriebene Zahlen genügen.",
            "false": "Etwas, das einen Beleg braucht, steht nicht in `fakten`, oder die "
            "sprechende Person sagt etwas über sich selbst über den Moment hinaus.",
        },
    }


# Variants sent with the production state (one request, answered independently).
FULL_STATE = {"old": old, "content": content, "structured": structured, "adds": adds}
# Variants sent in a second request whose state has no `fakten` and no `kontext`.
REDUCED_STATE = {"old_reduced": old, "content_reduced": content}
INVERTED = {"adds"}
# Round 2, production state, one request.
ROUND2 = {"facts": facts, "facts_structured": facts_structured, "structured": structured}
