# Wikipedia-Podcast-Pipeline – Designentscheidungen

Podcast-Name: **Gut zu wissen** (ohne „Wikipedia“ im Namen: Wikimedia-Marke; Wikipedia nur als Quellenangabe).

## Ziel

Aus einem Artikel der deutschen Wikipedia entsteht ein ca. 15-minütiger deutscher Podcast (Richtwert, kein Gate), in dem zwei feste KI-Personas sprechen. Faktische Behauptungen müssen aus dem Korpus belegt sein (Garantieniveau siehe unten).

## Anforderung Faktentreue

- **Korpus:** Hauptartikel plus Einleitungen verlinkter Artikel (siehe unten).
- **Belegpflichtig:** faktische Behauptungen, Meinungen, Prämissen in Fragen.
- **Verboten:** Persona-Aussagen über sich selbst (Erlebnisse, Biografie), ungeprüftes Allgemeinwissen.
- **Belegfrei:** Reaktionen und Überleitungen.
- **Garantieniveau:** probabilistisch. Falsch-Akzeptanzen, die Hörern nicht auffallen, werden toleriert.

## Pipeline

```mermaid
flowchart TD
    T[Auslöser: manuell oder Artikel des Tages] --> A[Abruf Wikitext, revid + UTC-Zeitstempel]
    A --> GP{Gate: lebende Person}
    GP -- ja --> FP[Manuell: Abbruch mit Hinweis / Artikel des Tages: Ersatzartikel]
    GP -- nein --> S[Segmentierung SoMaJo, Markup maskiert]
    S --> X[Extraktion pro Abschnitt]
    X --> G1{Gate: Segment-Regeln}
    G1 --> R[Ranking nach Überraschung: obere 50 % (D8)]
    R --> L[Einleitungen verlinkter Artikel]
    L --> X2[Extraktion + Gate + Ranking: 6–12 Fakten]
    S --> E[Einordnung: erste Einleitungssätze + Gate, bis 2]
    E --> D
    R --> D[Dialogautor]
    X2 --> D
    D --> G2{Gate: Fakt-IDs existieren}
    G2 --> J[Jev pro Redebeitrag: eine Frage pro Behauptung/Prämisse + Vollständigkeitsfrage]
    J -- "alle p ≥ t" --> M[Gate: kein Markup im Skript]
    J -- "ein p ≤ 1−t" --> JB[LLM-Judge: Ablehnung begründen]
    JB --> F[Fehlerkette]
    J -- "sonst, Überlänge oder vorübergehender Fehler" --> J2[LLM-Judge: Urteil]
    J -- Konfigurations- oder Anfragefehler --> AB[Abbruch, Eskalation an dich]
    J2 -- nicht belegt --> F
    J2 -- belegt --> M
    M --> TTS[Dialog-TTS Gemini 3.8 Flash TTS, Abschnitte an Redebeitrag-Grenzen]
    TTS --> STT[STT Whisper lokal, normalisierter Diff]
    STT --> H{Zweites Hören der Stelle: wie im Skript?}
    H -- ja --> P[Shownotes, Veröffentlichung]
    H -- nein --> Q{LLM: Aussage verändert?}
    Q -- ja --> F
    Q -- nein --> P
    F --> FT[Neugenerierung, Eskalation oder Ersatzartikel: siehe Fehlerbehandlung]
```

## Gate: lebende Personen

- **Zweck:** Risiko falscher Tatsachenbehauptungen über lebende Personen senken (Persönlichkeitsrecht, Haftung als Verantwortlicher nach § 18 Abs. 2 MStV).
- **Regel:** Artikel ist Personenartikel (Kategorie „Mann“, „Frau“, „Divers“ oder „Geschlecht unbekannt“) und hat keine Gestorben-Kategorie → lebend. Gestorben-Kategorien mit unscharfem Datum (z. B. „Gestorben im 20. Jahrhundert“) gelten als gestorben. Unklare Fälle gelten als lebend.
- **Prüfung:** deterministisch auf dem abgerufenen Wikitext, kein zusätzlicher API-Aufruf.
- **Geltung:** Hauptartikel und verlinkte Artikel. Verlinkte Artikel über lebende Personen werden aus dem Korpus ausgeschlossen; Fakten, die nur auf sie verlinken, bleiben.
- **Folge Hauptartikel:** manueller Lauf → Abbruch mit Hinweis; Artikel des Tages → Ersatzartikel. Ersatzartikel werden vor der Auswahl gegen das Gate geprüft.
- **Restrisiko:** Sachartikel, die lebende Personen erwähnen (Ereignisse, Unternehmen, Bands), passieren das Gate.

## Korpus

- **Hauptartikel:** roher Wikitext, vollständig.
- **Verlinkte Artikel:** nur Einleitung (Text vor der ersten `==`-Überschrift), nur für Links in den angebotenen Fakten des Hauptartikels (höchstens `linked_targets_max`). Angeboten werden aus ihnen nur Fakten, die direkt mit dem Hauptthema verbunden sind.
- **Quellstand:** revid aus der API-Antwort direkt gespeichert, zusätzlich UTC-Zeitstempel des Abrufs.

## Fakten

- **Segmentierung:** Wikitext vorab mit SoMaJo in Sätze zerlegt; `[[…]]`, Vorlagen und `<ref>`-Tags vor der Segmentierung maskiert, danach zurückgesetzt.
- **Fakt:** genau ein Segment, wörtlich inkl. Markup (Rohtext = Identität für Gates, Eindeutigkeit, Duplikate).
- **Klartext:** Jeder angebotene Fakt und sein Kontext werden bei der Extraktion per MediaWiki-API (`action=parse`, im Kontext des Artikels) gerendert; Klartext ohne `<ref>`-Tags wird neben dem Rohtext gespeichert. Inhaltstragende Vorlagen (z. B. Lebensdaten, Zahlenformate) bleiben so erhalten.
- **Gate Satzsegment:** eindeutig im eigenen Artikel, mindestens 8 Wörter, vollständiger Satz.
- **Gate Infobox-Zeile:** eigener Segmenttyp; nur eindeutig im eigenen Artikel.
- **Segment:** kleinste Einheit mit ID; Satzsegment (SoMaJo) oder Infobox-Zeile. Markup ist Teil des Segments.
- **Kontext:** lückenlose Vorgänger-Segmente, vom Extraktions-LLM gewählt, Anweisung „so gering wie möglich“. Kontext dient ausschließlich zur Auflösung (Pronomen, Ortsangaben), nie als Beleg.
- **Gate Kontext:** lückenlos, nur im eigenen Chunk, höchstens 3 Segmente. Verletzung → Fakt verworfen.
- **Chunking:** pro Abschnitt; große Abschnitte zusätzlich nach Segmentanzahl geteilt. Exakte Duplikate über Segment-ID ausgeschlossen; inhaltliche Dopplungen löst das Ranking.

## Ranking und Kontingent

- **Kriterium:** nur Überraschung (LLM, probabilistisch); bei verlinkten Einleitungen nur unter Fakten, die direkt mit dem Hauptthema verbunden sind.
- **Kontingent (D8, 2026-11-21):** Hauptartikel: die oberen `main_offer_share` (50 %) der extrahierten Fakten, aufgerundet, mindestens `main_facts_min` (20, sofern die Extraktion so viele liefert), höchstens `main_facts_max` (70; Entscheidung 2026-11-21 nach dem Kölner-Dom-Lauf mit 379 angebotenen Fakten); das LLM ordnet alle Fakten nach Überraschung (Dopplungen weit hinter dem überraschenderen), das Programm schneidet deterministisch ab; vom Modell nicht genannte Fakten folgen am Ende in Artikelreihenfolge. 6–12 Fakten verlinkte Einleitungen (eigenes Ranking, nur Fakten direkt mit dem Hauptthema verbunden: über das Hauptthema selbst oder seine direkte Beziehung zum verlinkten Gegenstand, nicht über den verlinkten Gegenstand allein). Liefert die Extraktion weniger als das Minimum, wird angeboten, was da ist. Angebot, keine Pflicht, mit Ausnahme der obersten `must_use_share` (5 %, aufgerundet, höchstens `must_use_max` = 15) des Hauptartikel-Rankings (Gate Top-Fakten, Dialog).
- **Einordnung:** die ersten Sätze der Einleitung des Hauptartikels (Text vor der ersten Überschrift), in Artikelreihenfolge, bis zu `einordnung_facts_max` (2), werden immer angeboten, außerhalb des Rankings und des Kontingents. Nur die deterministischen Fakt-Gates (eindeutig, Mindestwörter, vollständiger Satz); ein Satz, der eines verfehlt, wird übersprungen. Keine Infobox-Zeilen. Der erste Prosasatz der Einleitung (meist die Definition) ist immer Einordnung Nr. 1, sofern er die Gates besteht (Entscheidung 2026-11-20); ihn schließt D6 nie aus. Für die übrigen gilt der Ausschluss nach D6 wie bei anderen Fakten. Einordnungssätze werden nie zusätzlich gerankt oder als Hauptfakten angeboten; ein Segment, das auch extrahiert wurde, steht nur als Einordnung im Angebot. Im Angebot zuerst, mit Vermerk „(Einordnung)“; das Gespräch beginnt mit einer kurzen Einordnung daraus (was, wann, wo).
- **Schätzung:** 15 Minuten ≈ 2.000–2.300 Wörter, ein Fakt ≈ 100–150 Wörter → 15–20 Fakten.
- **Nachjustieren:** nach ersten Folgen anhand „verwendete Fakten pro Minute“.

## Dialog

- **Personas:** fest über alle Folgen; Expertin und neugierig Fragender. Persona-Dokument: Name, Stimme, Sprechstil, typische Reaktionen.
- **Erzählbogen:** wählt der Dialogautor aus dem Angebot.
- **Einheit:** Redebeitrag (bis Sprecherwechsel).
- **Ausgabe pro Redebeitrag:** Text, Fakt-IDs, jede Behauptung und jede Prämisse einer Frage als explizite Aussage mit Fakt-IDs. Reaktionen ohne Fakt-IDs zulässig.
- **Markup:** Dialogautor sieht nur Klartext von Fakten und Kontext und schreibt Klartext.
- **Gate Wiederholung:** ein Fakt steht in den Aussagen von höchstens `max_turns_per_fact` (2) Redebeiträgen (Erzählung plus eine Nachfrage); der letzte Redebeitrag (Zusammenfassung) zählt nicht. Jeder weitere Redebeitrag ist ein Gate-Problem und wird ohne Judge neu generiert; geprüft im ganzen aktuellen Dialog, auch nach jeder Neugenerierung, so trifft es immer den späteren Redebeitrag. Ausnahme (Entscheidung 2026-11-20): Felix' Redebeiträge, die die feste Einordnung Nr. 1 (den ersten Prosasatz der Einleitung) zitieren, meist als Prämisse einer Frage, zählen für diesen Fakt nicht und werden nicht geprüft; Lenas Redebeiträge zählen wie bei jedem Fakt (einmal erzählt, dazu die Zusammenfassung). In der Liste der schon erzählten Fakten für den Dialogautor steht der Satz mit dem Vermerk, dass Felix ihn erneut als Prämisse verwenden darf.

- **Gate Top-Fakten (D8, 2026-11-21):** die obersten `must_use_share` (5 %, aufgerundet; bei 131 extrahierten Fakten 7; höchstens `must_use_max` = 15, Entscheidung 2026-11-21) des Hauptartikel-Rankings müssen im Gespräch vorkommen, d. h. in den Aussagen angenommener Redebeiträge zitiert sein. Nicht verlangt werden Fakten, die D6 ausgeschlossen hat (kumulativ); der Rest des Rankings rückt nicht nach. Einordnung und verlinkte Fakten zählen nicht. Verfehlt der Dialogautor das Gate, gilt wie bei D5 Gesprächsneugenerierung (Grund: die fehlenden Fakten); nach dem Judge ein Gate-Fehlschlag.

## Judge

- **Jev:** Entscheidungsmodell von TypeSafe (`jev-1.14.0`, Version fest angegeben, kein Alias; Auswahl siehe Spike S1), Zugang direkt über TypeSafe; liefert pro Frage eine Wahrscheinlichkeit p, keinen Text. p ist laut Anbieter kalibriert, unabhängig nicht bestätigt; die Stichproben prüfen das. Limit: 64k Token gesamt, 32k für State plus längste Frage.
- **LLM-Judge:** Claude Sonnet 5.5 (dasselbe Modell wie der Dialogautor), ohne geteilten Kontext mit dem Dialogautor. Eingabe (geändert 2026-11-21, #8, #15): vorheriger Redebeitrag (kein Beleg), Redebeitrag, Fakt-Sätze, Kontext; beim Urteil nach unsicherem Jev-Ergebnis zusätzlich die Jev-Fragen mit p < t als Hinweis (Aussagen nur über ihre Fakt-IDs genannt). Anbieter austauschbar halten.
- **Eingabe (Jev-State):** Redebeitrag, Klartext der zitierten Fakt-Sätze, Klartext des Kontexts als getrenntes Feld.
- **Rubrik:** jede Behauptung und Prämisse folgt aus Fakt-Sätzen. „Nur durch Kontext belegt“ gilt als nicht belegt. Redebeitrag ohne Fakt-IDs darf keine Behauptung enthalten. Reaktionen nach dem Reaktionskriterium im Persona-Dokument gelten als belegfrei. Jede Aussage meint im Anschluss an den vorherigen Redebeitrag denselben Gegenstand wie ihr Fakt-Satz (2026-11-21, #8, #15): Bezugswort ohne genannten Gegenstand, Antwort mit einem Fakt zu einem anderen Gegenstand als die Frage, Verbindung von Fakten, die kein Fakt-Satz nennt → nicht belegt. Den Gegenstand eines Bezugsworts mit Namen zu nennen, wie der Kontext ihn bestimmt, ist Auflösung, kein Beleg.
- **Fragen:** eine Jev-Frage pro Behauptung und Prämisse; zusätzlich eine Vollständigkeitsfrage „keine weiteren Behauptungen“. Deren Anweisung enthält das Reaktionskriterium.
- **Schwellwert t:** ein Wert für alle Fragen, symmetrisch:
  - alle p ≥ t → belegt.
  - mindestens ein p ≤ 1−t → nicht belegt. LLM-Judge liefert nur die Begründung („begründe die Ablehnung“), urteilt nicht. Neugenerierung folgt immer.
  - sonst → LLM-Judge prüft den ganzen Redebeitrag mit voller Rubrik, urteilt und begründet.
- **Länge:** hartes Limit prüft die API; Überlänge → LLM-Judge. Weiche Grenze in Zeichen, erst nach Kalibrierung.
- **API-Fehler:**

| Fehlerart | Codes | Reaktion |
|---|---|---|
| Überlänge | 400 mit `error_type` `max_tokens_exceeded` | LLM-Judge |
| Vorübergehend | 429, 500, 529, Timeout | Wiederholen mit Wartezeit, danach LLM-Judge |
| Konfiguration | 401, 402, 403, 404; 400 mit `error_type` `api_usage_error` (z. B. unbekanntes Modell) | Abbruch |
| Ungültige Anfrage | 400 (außer Überlänge), 413, 415, 422 | Abbruch |

## Fehlerbehandlung

| Stufe | Manueller Lauf | Artikel des Tages |
|---|---|---|
| Redebeitrag | max. 2× neu nach Judge-Ablehnung, max. 3× nach Gate-Ablehnung, Nachbarn als Kontext | max. 2× neu nach Judge-Ablehnung, max. 3× nach Gate-Ablehnung, Nachbarn als Kontext |
| Gespräch | max. 2× neu: ab dem auslösenden Redebeitrag fortgesetzt, sonst ganz neu | max. 2× neu: ab dem auslösenden Redebeitrag fortgesetzt, sonst ganz neu |
| Audio | betroffene Redebeiträge je max. 2× neu synthetisieren | betroffene Redebeiträge je max. 2× neu synthetisieren |
| Danach | Eskalation an dich | Ersatzartikel |
| Ersatzartikel | – | zufällig aus exzellenten und lesenswerten Artikeln, max. 2×, danach entfällt die Folge |
| Jev-Abbruch | Eskalation an dich | Eskalation an dich, Folge entfällt |

**Budgets Redebeitrag (D1, geändert 2026-11-20):** pro auslösendem Redebeitrag zwei getrennte Budgets. Judge-Ablehnungen (Jev oder LLM-Judge) max. 2× (`turn_retries`), Gate-Ablehnungen (ohne Judge-Aufruf: unbekannte Kennung, Wiederholung, Markup, leerer Text, Aussage ohne Kennungen, unbekannte Person) max. 3× (`gate_retries`). Eine Neugenerierung nach einer Ablehnung belastet das Budget dieser Ablehnung; die Kaskadenschritte belasten das Budget der Ablehnung, die die Kaskade ausgelöst hat; wird ein Kaskadenschritt abgelehnt, belastet seine Neugenerierung das Budget dieser Ablehnung. Jede Neugenerierung vermerkt das belastete Budget.

**Gespräch neu (D6, geändert 2026-11-20):** Fakten, die abgelehnte Aussagen zitieren, werden ausgeschlossen (kumulativ). Ist das Budget eines Redebeitrags erschöpft, bleiben die Redebeiträge vor dem auslösenden Redebeitrag k mit ihren Urteilen erhalten (k: der auslösende Redebeitrag, wenn es in einer Kaskade scheitert, sonst der gescheiterte). Der Dialogautor schreibt das Gespräch ab k bis zum Ende weiter, mit der einen Zusammenfassung, auf demselben Angebot ohne die ausgeschlossenen Fakten (kein neues Ranking); er sieht die behaltenen Redebeiträge und die darin schon erzählten Fakten und erzählt sie nicht noch einmal. Er erhält den Grund der letzten Ablehnung der gescheiterten Kette (die das Budget erschöpft hat; frühere Ablehnungen der Kette kurz) in seiner Form (Kurzkennungen F1, F2 … statt Segment-IDs, deutsche Gate-Texte) und vermeidet diesen Fehler (Entscheidung 2026-11-20). Geprüft werden nur die neuen Redebeiträge; Gate Wiederholung und die Gates nach D5 gelten für das ganze Gespräch. Neues Ranking und neues Gespräch nur, wenn k = 0 ist, das verbleibende Angebot das Minimum nach D5 nicht erreichen kann oder D5 nach dem Dialogautor verfehlt wird. Zählt gegen das Gesprächsbudget (max. 2×). Der verworfene Teil (ab k oder das ganze Gespräch) wird mit dem gewählten Weg archiviert, bei einer Fortsetzung mit dem übergebenen Grund.

**Eskalation an dich:** Workflow-Lauf schlägt fehl (GitHub-Standardbenachrichtigung); Grund in Log und Job-Zusammenfassung. Ein erfolgreicher manueller Lauf gilt ohne aktive Freigabe als freigegeben.

Alle Textversionen werden archiviert (inkl. STT-Transkripte und Diff-Entscheidungen verworfener Synthesen). Audio nur in der veröffentlichten Fassung.

## Audio

- **Vorbereitung:** Gate prüft, dass das Skript kein Wiki-Markup enthält; Fund → Fehler des Redebeitrags (Neugenerierung).
- **Synthese:** Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`; Multi-Speaker-Dialog mit höchstens zwei Sprechern und vordefinierten Stimmen). Skript (ca. 14.000–16.000 Zeichen) in Abschnitte an Redebeitrag-Grenzen geteilt, höchstens 6.000 Zeichen (rund sechs Minuten Audio); Audio zusammengefügt. Die dokumentierten Limits (8.192 Token Eingabe, 16.384 Token Ausgabe) werden eingehalten, die API setzt sie aber nicht durch; die Abschnitte dienen dazu, einzelne Teile günstig neu zu synthetisieren (Spike S4).
- **Ratenlimit:** 10 Anfragen pro Minute (Tier 1). Bei `429` wartet der Client die in der Antwort genannte Zeit ab.
- **Lautheit:** Normalisierung auf −16 LUFS mit True-Peak-Limiter (Decke −1,5 dBTP). Die Synthese liefert rund −20 LUFS; ohne Limiter würde die Verstärkung die Spitzen übersteuern.
- **Verifikation:** Whisper lokal im Actions-Runner (Modellgröße nach Messung), Diff gegen bereinigtes Skript; vorher normalisiert (Zahlwörter, Satzzeichen, Groß-/Kleinschreibung).
- **Abweichung:** zuerst zweites Hören; bestätigt es die Abweichung, entscheidet das LLM, ob sie die Aussage ändert; nur dann Neusynthese der betroffenen Redebeiträge (siehe Fehlerbehandlung).
- **Zweites Hören (Entscheidung 2026-11-21, vor das LLM gezogen am selben Tag, spart Tokens):** Whisper verhört sich zufällig. Jede Abweichung wird über Whispers Wortzeiten im Audio verortet, mit einigen Skriptwörtern davor und danach ausgeschnitten und noch einmal transkribiert. Hat dieses Transkript an der Stelle keine Abweichung, gilt sie als Hörfehler, zählt nicht und geht nicht ans LLM. Beide Transkripte werden archiviert.
- **Neusynthese einzelner Redebeiträge (Entscheidung 2026-11-21):** Nicht der ganze Abschnitt wird neu synthetisiert, nur die Redebeiträge mit den verbliebenen Abweichungen. Das neue Audio ersetzt das alte zwischen den leisesten Stellen an den Grenzen zum vorigen und nächsten Redebeitrag (Wortzeiten von Whisper); Randstille gekürzt, eine kurze Pause dazwischen. Geprüft wird nur das neue Audio. Lässt sich eine Grenze nicht schneiden (keine Wortzeiten, überlappende Wörter), wächst der neu synthetisierte Teil bis zur nächsten schneidbaren Grenze, höchstens bis zum ganzen Abschnitt. Jeder Redebeitrag wird höchstens 2× neu synthetisiert (`audio_section_retries`), danach Eskalation.
- **Verschluckte Kleinstwörter (Entscheidung 2026-11-20):** Hört Whisper ein kleines Wort oder eine Endung nur verschluckt oder zusätzlich („Zeigt dir“ statt „Zeigte“, „Grieche“ statt „Griechisch“) und behält der Satz seinen Sinn, ändert das die Aussage nicht. Zahlen, Daten, Namen, Verneinungen und Wörter mit anderer Bedeutung bleiben streng; im Zweifel gilt die Aussage als verändert.
- **Aussprache-Lexikon (Entscheidung 2026-11-20):** Für Namen, die die Stimme falsch spricht oder die Whisper in üblicher Schreibung statt wie gesprochen schreibt. Einträge `written`, `spoken`, `heard`: `spoken` geht statt `written` nur an die TTS, `heard`-Varianten zählen im Transkript als `written` (außer sie stehen selbst im Skript). Skript und Archiv behalten die Originalschreibung. Das allgemeine Lexikon steht in der Config (`[[pronunciation]]`, gilt für alle Folgen; derzeit nur „Lena“ mit den Hörvarianten „Lina“, „Leda“); Einträge für eine einzelne Folge stehen in `pronunciation.json` im Folgenordner (von Hand gepflegt, Folgeneinträge haben Vorrang). Das gültige Lexikon steht in `08_audio.json`.

## Feedback und Kalibrierung

- **Schwellwert:** Startwert t = 0,95, danach nachkalibriert. Getrennte Werte pro Fragetyp erst, wenn Daten es rechtfertigen.
- **Weiche Längengrenze:** Fehlerrate von Jev über State-Länge in Zeichen; Grenze dort, wo sie steigt.
- **Labelquellen:** LLM-Judge-Urteile auf Jev-Eskalationen (automatisch); deine Stichproben (siehe unten); Hörer-Feedback.
- **Stichproben:**
  - Grundgesamtheit: angenommene Redebeiträge mit Fakt-IDs (durch Jev oder LLM-Judge angenommen).
  - Ziehung: jeder Redebeitrag mit Wahrscheinlichkeit gleich der Rate r.
  - Ablauf: Pipeline legt Review-Issue an; du kommentierst pro Redebeitrag maschinenlesbar „belegt“ oder „nicht belegt“; ein Folgeschritt schreibt die Labels ins Archiv. Ungeprüfte Redebeiträge gelten als unbekannt.
  - Rate: Start r = 100 %, Untergrenze 1 %. Fehlerfrei geprüfte Folge → r × 2/3. Gefundener Fehler → r × 2, höchstens 100 %. Nur beantwortete Issues zählen; Folgen ohne gezogenen Redebeitrag ändern r nicht.
- **Hörer-Feedback:** E-Mail an gut-zu-wissen@fourteen-it.de. Shownotes enthalten pro Folge einen mailto-Link mit Folgentitel und Folgen-ID im Betreff. Zweck: allgemeines Feedback und Fehlerhinweise. Du liest die Mails persönlich, kein Pipeline- oder LLM-Zugriff aufs Postfach; Hörerdaten bleiben im Postfach. Bestätigter Fehler → du legst ein Issue im Review-Format an; zählt als Label und verdoppelt die Stichprobenrate (höchstens 100 %).
- **Review:** direkt in den Archivdateien. Pro Redebeitrag lesbar: Text, Fakt-IDs, Fakt-Sätze, Kontext, Judge-Urteil, Konfidenz, Begründung.

## Lizenz

- Podcast unter CC BY-SA 4.0.
- Repo: Code unter Apache 2.0 (`LICENSE` im Wurzelverzeichnis); Archiv unter CC BY-SA 4.0 (eigene Lizenzdatei im Archiv-Verzeichnis); README nennt die Zuordnung. Dateien mit Wikipedia-Text (z. B. Test-Fixtures) liegen getrennt mit CC-BY-SA-Hinweis.
- Namensnennung in Shownotes (Revisions-Permalinks für Hauptartikel und jeden verwendeten verlinkten Artikel, Link zur Versionsgeschichte) und ein Satz im Audio-Abspann.
- Juristische Prüfung bei Monetarisierung.
- Anbieter: FOURTEEN IT. Eigene Rechtstexte auf der Pages-Seite (impressum.md, datenschutz.md), auf jeder Seite verlinkt. Impressum inkl. Verantwortlicher nach § 18 Abs. 2 MStV.
- Anforderungen an die Pages-Seite aus der Datenschutzerklärung: keine Cookies, kein Tracking, keine Drittanbieter-Ressourcen (z. B. externe Schriften); Audio-Player lädt erst bei Wiedergabe (`preload="none"`).

## Begrüßung und Abspann

Fester Text außerhalb des Dialogautors und der Persona-Regeln; einmal synthetisiert, Audiodateien im Repo, in jeder Folge wiederverwendet; nicht Teil des STT-Loops.

**Begrüßung:**
- Felix: „Hallo und willkommen bei ‚Gut zu wissen‘!“
- Lena: „In jeder Folge nehmen wir uns einen Wikipedia-Artikel vor und präsentieren euch die interessantesten Fakten.“

**Abspann:**
- Lena: „Das war ‚Gut zu wissen‘. Alles, was wir erzählt haben, stammt aus der deutschsprachigen Wikipedia. Die Artikel und ihre Autorinnen und Autoren findet ihr in den Shownotes.“
- Felix: „Ein wichtiger Hinweis: Diese Folge ist mit KI entstanden – von der Auswahl der Fakten über das Gespräch bis zu unseren Stimmen. KI kann Fehler machen. Und die Wikipedia-Inhalte selbst sind nicht unabhängig geprüft. Wenn ihr euch auf etwas verlassen wollt, prüft es bitte nach.“
- Lena: „‚Gut zu wissen‘ steht unter der Lizenz C C B Y S A 4.0 und ist ein unabhängiges Projekt, nicht verbunden mit der Wikimedia Foundation. Bis zum nächsten Mal!“ (Buchstaben mit Leerzeichen, damit die Sprachausgabe „CC BY-SA 4.0“ buchstabiert.)

## Betrieb

- **Auslöser:** manuell auf Abruf (`workflow_dispatch` mit Artikel als Eingabe); automatisch täglich mit Artikel des Tages (`schedule`, `17 3 * * *` UTC; Ziel: Folge bis ca. 6 Uhr deutscher Zeit; Start nach dem Wechsel des Artikels des Tages, unabhängig davon, ob er nach UTC oder deutscher Zeit wechselt).
- **Automatische Veröffentlichung (Entscheidung 2026-11-20):** Die Tagesfolge wird einmal täglich zu einer festen Uhrzeit (10:00 UTC, Workflow `auto-publish.yml`) veröffentlicht, wenn ihr Lauf mindestens `min_age_hours` (Start 2 Stunden) vorher endete; bis dahin kannst du eingreifen. Eine später fertige Folge wartet bis zum nächsten Tag. Bremse: Label `hold` am Review-Issue (Label entfernen gibt frei). Nie automatisch: Folgen, deren Review einen Fehler fand, und Folgen ohne `audio_run_id`; sie veröffentlichst du von Hand (`publish.yml`). Ein verworfener Tageslauf oder ein Absturz lässt den Workflow-Lauf scheitern; GitHub schickt dir dazu eine Mail.
- **Artikel des Tages (Spike S8):** Chronologie-Seite `Wikipedia:Hauptseite/Artikel des Tages/Chronologie <Jahr>` (Bot, 23:00 UTC für den Folgetag), Wochentagsseite als Rückfall. Ersatzartikel aus `Kategorie:Wikipedia:Exzellent` und `Kategorie:Wikipedia:Lesenswert`, zufällig mit dem Datum als Seed.
- **Sprache:** Python.
- **Orchestrierung:** GitHub Actions in einem öffentlichen Repo (unbegrenzte Minuten). API-Keys als Secrets. Workflow-Logs, Prompts und Archiv sind öffentlich.
- **Archiv:** Text als JSON im Repo (eingerückt, stabile Feldreihenfolge). Audio nur beim Hoster.
- **Archivlayout:** `archive/episodes/<Datum des Laufs>_<Slug>/` pro Lauf (Slug: Kleinbuchstaben, Umlaute transliteriert, sonst `-`; am selben Tag Suffix `-2`, `-3`). Darin `episode.json` (Gesamtstatus `running`/`completed`/`published`/`failed`, Titel, Zeitstempel, `published_at`, `explicit`, Fehlergrund) und eine Datei pro Pipeline-Schritt: `01_article.json`, `02_segments.json`, `03_facts.json`, `04_ranking.json`, `05_dialog.json`, `06_judgement.json`, `07_explicit.json`, `08_audio.json` (Sektionen samt STT-Prüfung: Transkript, Abweichungen, Entscheidungen, jeder Syntheseversuch), `09_review.json` (Stichprobe samt Seed und Ziehungen, alles zur Beurteilung der gezogenen Turns, Labels aus dem Review-Issue); dazu `linked/<slug>.json` für verlinkte Einleitungen. Fehlgeschlagene Läufe werden mit `status: failed` committet. Das Wiederholungs-Gate liest `published_at` der veröffentlichten Folgen mit gleichem kanonischem Titel. Stichprobenrate r: `archive/state/review.json`.
- **Kosten:** kein Limit, Protokoll pro Folge.
- **Wiederaufnahme ab Audio:** Scheitert ein Lauf nur im Audioschritt (TTS- oder STT-Fehler, STT-Budget erschöpft, Absturz im Schritt), lässt er sich ab dem Audioschritt wiederaufnehmen (Workflow `resume-audio.yml`): archiviertes finales Skript, derselbe Audioschritt, danach Stichprobe und Review-Issue; jede andere Folge wird abgelehnt. Die Folge zählt als ein Lauf mit ihrem Endstatus; Laufzeit und Kosten umfassen alle Versuche (Entscheidung 2026-11-20).

## Anbieter

| Aufgabe | Anbieter |
|---|---|
| Extraktion (Hauptartikel und Einleitungen verlinkter Artikel) | Claude Haiku 5.5 |
| Ranking, Dialogautor | Claude Sonnet 5.5 |
| Judge Stufe 1 | Jev, TypeSafe direkt |
| Judge Stufe 2 (Urteil, Begründung) | Claude Sonnet 5.5 |
| TTS | Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`), vordefinierte deutsche Stimmen |
| STT | Whisper lokal |
| Audio-Hosting | Internet Archive |
| Feed, Shownotes | GitHub Pages aus dem Pipeline-Repo |
| Verbreitung | Spotify (RSS-Feed eingereicht; Spotify hat keine Upload-API) |

## Ausbaustufen

1. Manueller Lauf bis Audio, ohne Veröffentlichung.
2. Veröffentlichung: Internet Archive, RSS-Feed, Shownotes.
3. Artikel des Tages, täglicher Auslöser, Ersatzartikel.

## Offene Punkte

- **Whisper:** Start mit großem Modell, Laufzeit im Runner messen; Modell im Actions-Cache.
- **Rechtstexte:** ⚠-Platzhalter in datenschutz.md klären, juristische Prüfung empfohlen (vor Ausbaustufe 2).

## Bekannte Risiken

- **Falsch-Akzeptanzen:** nur über Stichproben und Hörer-Feedback sichtbar; Schwellwert kann zwischen Stichproben unbemerkt driften.
- **Prämissen:** erhöhen voraussichtlich die Eskalationsrate an den LLM-Judge.
- **Kontext:** Obergrenze begrenzt Segmentanzahl, nicht Länge; lange Jev-States können Jevs Genauigkeit senken, bis die weiche Grenze kalibriert ist.
- **Begründungsauftrag:** erzwingt eine Begründung; bei falscher Jev-Ablehnung entfernt die Neugenerierung eine korrekte Behauptung (Inhaltsverlust, kein Faktenfehler).
- **Jev-Schwächen:** laut TypeSafe Zahlen, Zählen, Datumsvergleiche, Mehrschritt-Fragen, großer State. Deutsch nicht dokumentiert.
- **Eskalationsrate:** bei t = 0,95 anfangs voraussichtlich hoch.
- **Judge-Modell:** LLM-Judge und Dialogautor sind dasselbe Modell; korrelierte Fehler möglich. Messung: Übereinstimmung mit Stichproben und Hörer-Feedback; bei schwacher Übereinstimmung Anbieter wechseln.
- **TTS-Abschnitte:** Prosodie, Lautstärke und Stimme können an Übergängen springen; STT prüft nur den Wortlaut.
- **Audio-Archiv:** fällt das Internet Archive weg, ist das Audio weg; Neusynthese aus dem Text-Archiv nicht identisch.
- **Whisper-Laufzeit:** auf CPU ungeprüft; Grenze ist das Job-Timeout.
- **Rendern außerhalb des Kontexts:** einzelne Vorlagen können beim Rendern einzelner Segmente anders ausfallen als im Artikel; in Ausbaustufe 1 prüfen.
- **Lebende Personen:** Gate erfasst nur Personenartikel; Erwähnungen lebender Personen in Sachartikeln bleiben möglich. Fehlkategorisierte Artikel passieren das Gate.
- **Freie Reaktionen:** liegen näher an der Grenze zur Behauptung; mehr unsichere Urteile bei der Vollständigkeitsfrage und mehr Eskalationen erwartet.
- **Kalibrierungsdaten:** menschliche Labels nur aus Stichproben und Hörer-Feedback; bei Untergrenze 1 % wenige Labels pro Monat.
- **Unbeantwortete Review-Issues:** senken die Rate nicht, liefern aber auch keine Labels.
- **Hörer-Feedback:** abhängig von deiner manuellen Bearbeitung; unbearbeitete Mails liefern keine Labels.
