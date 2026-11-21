# Issues #8 und #15: Annahmen und offene Punkte

Stand: 2026-11-21, alle Annahmen vom Owner bestätigt. Gebaut nach deiner Vorgabe „Lösungen eigenständig implementieren, Annahmen sammeln“. Jede Annahme hier kannst du ändern.

## Umgesetzt

| Commit | Änderung | Issue |
|---|---|---|
| Judge: previous turn as input; rubric … | Der LLM-Judge sieht den vorherigen Redebeitrag. Die Rubrik lehnt drei Fälle ab: ein Bezugswort ohne genannten Gegenstand; eine Antwort, deren Fakt-Satz von einem anderen Gegenstand handelt als die Frage; eine Verbindung von Fakten, die kein Fakt-Satz nennt. Eine Frage, die ihre Antwort nahelegt, gilt als Prämisse. | #8; #15, Beiträge 26, 51, 53 |
| Judge: Jev questions below t as a hint … | Im Unsicher-Fall listet die Judge-Eingabe die Jev-Fragen mit p < t samt p als Hinweis, nicht als Urteil. Aussagen nennt sie nur über ihre Fakt-IDs. | #15, Frage 3 |
| Dialog author: name referents … | Drei Autorregeln: Bezugswort mit Namen aus dem Kontext auflösen; Antwort mit einem Fakt zum Gegenstand der Frage; Felix' Fragen legen keine Antwort nahe. | #8, #15 |

## Annahmen

1. **Lösung am Judge und am Autor, nicht an der Extraktion.** Von den drei Richtungen in #8 habe ich die dritte umgesetzt (Judge prüft den Bezug) und die zweite beim Autor statt bei der Extraktion (er löst den Bezug beim Sprechen auf). Fakten mit hängendem Bezug werden nicht aussortiert. Grund: Die Extraktion wählt Kontext nach Anweisung genau für solche Bezüge (Pronomen, Ortsangaben); Aussortieren träfe viele Fakten mit Kontext, etwa in #16 die Fossa Carolina (Beitrag 50) und das Faltboot-Hebewerk (Beitrag 46).
2. **Den Namen aus dem Kontext zu nennen gilt als Auflösung, nicht als Beleg aus dem Kontext.** Das deckt sich mit dem Designdokument („Kontext dient ausschließlich zur Auflösung“). Nur der Name ist erlaubt, keine weitere Angabe aus dem Kontext. Folge: Jevs Aussagefrage („was nur in `kontext` steht, gilt als nicht belegt“) kann bei solchen Aussagen unsicher werden, und mehr Beiträge gehen an den LLM-Judge.
3. **Der LLM-Judge sieht den vorherigen Redebeitrag.** Jev bekommt ihn seit S1. Das Designdokument nennt als Eingabe des LLM-Judges „nur Redebeitrag, Fakt-Sätze, Kontext“. Das „nur“ lese ich als Abgrenzung zum Dialogautor (keine Statements, keine Gedankenkette). Der vorherige Beitrag ist Gesprächstext und laut Prompt kein Beleg. Ohne ihn kann der Judge #15/51 nicht erkennen.
4. **Vorwegnahme in Fragen über die bestehende Prämissenregel.** Ich lese „Wertwort in der Frage“ als Prämisse, die schon jetzt Fakt-IDs braucht (`persona.md`: „stellt Prämissen nur, wenn sie belegt sind“). Deshalb steht die Regel in der Judge-Rubrik und im Autor-Prompt, `docs/persona.md` ist unverändert.
5. **Kein strengerer Schwellwert für Felix.** In #14 und #16 haben korrekte Felix-Fragen Vollständigkeits-p von 0,17 bis 0,29 (z. B. #14/63, /67, /75, #16/54). Ein strengerer Schwellwert hätte sie abgelehnt. Deshalb bekommt der Judge nur einen Hinweis, keinen zusätzlichen Schwellwert.
6. **Der Hinweis zeigt keinen Text des Autors.** Eine zweifelhafte Aussage wird über ihre Fakt-IDs genannt, nicht über den Aussagetext, damit der Judge keinen Kontext mit dem Autor teilt.
7. **Jev-Fragen unverändert.** Der Skill `typesafe-ai` (Pflicht laut `CLAUDE.md` für Fragenwortlaut) ist in dieser Sitzung nicht installiert. Außerdem prüft ein Test, dass die Vollständigkeitsfrage die in S1 gemessene ist. Jev erkennt hängende Bezüge also weiter nicht. Beiträge, die Jev mit p ≥ t annimmt, erreichen den Judge nicht (#8 und #15 betrafen nur LLM-Judge-Annahmen).
8. **Grenzfälle aus #15 (Beiträge 3, 22, 60) ohne eigene Regel.** „Wie ging die Geschichte schließlich aus?“ (60) fällt unter die Regel „Frage legt Antwort nahe“ nur, wenn der Judge „schließlich aus“ als Einzelheit liest. Eine eigene Regel habe ich nicht gebaut.

## Offen für dich

- **Dokumente:** Freigegeben am 2026-11-21 („Passt alles“) und so übernommen:
  - `docs/wikipedia-podcast-pipeline.md`, Judge, „LLM-Judge“: „Eingabe: vorheriger Redebeitrag (kein Beleg), Redebeitrag, Fakt-Sätze, Kontext; bei unsicherem Jev-Urteil die Jev-Fragen mit p < t als Hinweis.“
  - ebd., „Rubrik“: „Jede Aussage meint im Anschluss an den vorherigen Redebeitrag denselben Gegenstand wie ihr Fakt-Satz (Bezugswort ohne genannten Gegenstand, Antwort auf eine andere Frage, Verbindung, die kein Fakt nennt → nicht belegt).“
  - `docs/persona.md`, Reaktionskriterium, „Nicht erlaubt“: „Nachfrage, die eine Antwort nahelegt (Wertwort, Vermutung, Einzelheit aus einem späteren Fakt).“
- **Nicht gemessen:** Die neuen Prompts sind nur mit Fakes getestet. Ob der Judge #8/32 und #15/26, 51, 53 jetzt ablehnt, zeigt erst ein Lauf oder ein Replay auf den archivierten Folgen (kostet API-Aufrufe). Ich habe nichts live aufgerufen.
- **Kosten und Eskalationen:** Mehr Ablehnungen bedeuten mehr Neugenerierungen und Kaskaden (D1). Wie viel mehr, zeigt `python -m gut_zu_wissen.metrics` nach den nächsten Läufen.
- **Veröffentlichte Folgen:** #8 (Antikythera) und #15 (Galerie 291) enthalten die Fehler weiterhin. Korrekturen habe ich nicht angefasst.
- **#14 und #16:** Das sind Review-Issues ohne deine Labels. Die gezogenen Beiträge kann nur `ajFourteen` labeln.

## Erster Testlauf (2026-11-21, Kölner Dom, Lauf 44945018484)

- **Ergebnis:** abgebrochen, Gesprächsbudget aufgebraucht (3 Gespräche). LLM-Judge lehnte 23 von 143 Beiträgen ab; frühere Läufe 0–6 %.
- **Ursache:** Die neuen Regeln griffen zu weit. Offene Überleitungsfragen („Gibt es im Inneren etwas Ungewöhnliches?“) galten als Vorwegnahme; „der Dom“ galt als ungeklärter Bezug, weil der Judge den Artikeltitel nicht sah.
- **Nachschärfung (#22):** Die Judge-Eingabe nennt den Artikel jedes Fakts. Eindeutig ist ein Gegenstand, den der Redebeitrag oder der vorherige Beitrag nennt, und das Thema des Artikels. Eine Frage ist nur dann Prämisse, wenn sie eine bestimmte Einzelheit, Wertung oder Vermutung vorwegnimmt oder eine nicht gesagte Tatsache voraussetzt.
- **Weiter offen:** Das Wiederholungs-Gate (Fakt doppelt erzählt) verbrauchte in Gespräch 1 und 2 das Gate-Budget; das ist älter als diese Änderung.

## Drittes Hören mit Hinweis (2026-11-21, nach dem Lexikon-Eintrag für Kölner Dom)

Gebaut nach deinem „Baue es so wie von dir vorgeschlagen“. Annahmen:

1. **Hinweis = `hotwords` von faster-whisper** mit dem erwarteten Wort. Nicht `initial_prompt`, das Whisper stärker zum Nachschreiben verleitet.
2. **Ähnlichkeit = difflib-Verhältnis ≥ 0,65.** Rainald/Reinhard 0,67, Patrizier/Patricia 0,71 kommen durch, Gerhard/Konrad 0,46 nicht. Petersglocke/Kaiserglocke hat wegen der gemeinsamen Endung auch 0,67; dort entscheidet nur das dritte Hören. Nicht an weiteren Fällen gemessen.
3. **„Wort aus den Fakt-Sätzen“ statt „Name“.** Im Deutschen sind alle Substantive groß geschrieben, ein Name lässt sich nicht sicher erkennen; „Patrizier“ ist kein Name, war aber der zweite Fall.
4. **Ausgeschlossen:** Verneinungen und Monatsnamen (Juni/Juli 0,75), dazu alle Zahlen. Liste in `[stt] hint_excluded_words`.
5. **Höchstens 3 Wörter pro Folge**; danach zählen weitere Abweichungen wie bisher. Ein Wort mit zweiter Hörvariante zählt nicht doppelt.
6. **Die Pipeline schreibt `pronunciation.json` selbst**, auch bei Abbruch, damit ein Resume die Wörter kennt. Jeder Eintrag hat `note` mit dem Grund. Der Eintrag ist Archiv, also öffentlich und im Review sichtbar.
7. **Nicht gemessen:** wie oft Whisper mit Hinweis das erwartete Wort auch dann schreibt, wenn die Stimme es anders sagt. Der erste echte Fall zeigt es; prüfen lässt es sich nur durch Hören.
