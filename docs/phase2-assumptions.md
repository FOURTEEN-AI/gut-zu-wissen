# Phase 2: Annahmen und offene Fragen

Stand: 2026-11-20. Entstanden, während Andreas abwesend war. Die Quelldokumente (Design, Persona, Plan, Rechtstexte) sind bis auf die genannten Stellen unverändert. Alles hier ist eine Annahme von Claude, bis es bestätigt ist.

## Vor dem Go-live zu tun (nur Andreas)

1. ~~GitHub-Secrets~~ erledigt: `ARCHIVE_S3_ACCESS_KEY` und `ARCHIVE_S3_SECRET_KEY` existieren, dieselben Namen wie in `.env`.
2. ~~Pages aktivieren~~ erledigt (2026-11-20, per API): Quelle „GitHub Actions“, Adresse `https://fourteen-ai.github.io/gut-zu-wissen/` (bestätigt die Annahme zur Adresse).
3. **Erster Upload mit einer Testfolge**, bevor die echte Folge live geht. Der Upload ist nur gegen Fakes getestet (siehe unten).
4. **Spotify-Einreichung** (manuell, einmalig) mit der Feed-URL; die Bestätigungsmail geht an gut-zu-wissen@fourteen-it.de.
5. Die **Datenschutz-Aussagen** zu IONOS-Auftragsverarbeitungsvertrag und DPF-Zertifizierung von GitHub bestätigen.

## Annahmen zum Internet Archive (ungeprüft gegen den Live-Dienst)

- Die `x-archive-meta*`-Header gelten nur beim Anlegen eines Items; deshalb sendet `publish` sie nur, wenn `item_exists` falsch liefert.
- Der Metadaten-Endpunkt (`archive.org/metadata/<id>`) antwortet für ein nicht vorhandenes Item mit `{}` und Status 200.
- Die Dateiliste eines neuen Items zeigt die Dateien nach höchstens 15 Minuten (`ready_timeout_seconds`). Bis dahin gilt die Folge nicht als veröffentlicht.
- Sammlung `opensource_audio`, Medientyp `audio`, Derivate aus (`x-archive-queue-derive: 0`). Folge: Das Item hat keine automatische Vorschaugrafik und keine Streaming-Derivate.
- Die Download-URL `archive.org/download/<id>/<datei>` leitet auf einen Storage-Server weiter. Spotify und Apple folgen Weiterleitungen in der Regel. Falls eine Plattform den Feed ablehnt, wäre das die erste Stelle zum Suchen.
- Item-Kennung: `gut-zu-wissen_<Folgen-ID>`. Bei mehr als 100 Zeichen wird gekürzt und ein Hash angehängt.

## Annahmen zur Site und zum Feed

- **Adresse:** `https://fourteen-ai.github.io/gut-zu-wissen` (aus der User-Agent-Zeichenkette in `github.py` abgeleitet). Bei eigener Domain `[site] base_url` in der Config ändern; die Feed-URL ändert sich dann, und Spotify müsste neu eingereicht werden.
- **Folgentitel im Feed und auf der Seite:** der Artikeltitel (kein „Gut zu wissen:“-Präfix). Im MP3 und im Archive-Item steht dagegen „Gut zu wissen: <Titel>“.
- **Cover wird auf den Seiten nicht angezeigt**, weil es sonst vom Internet Archive geladen würde (Datenschutz: keine externen Ressourcen). Es steht nur im Feed (`itunes:image` pro Folge). Das Show-Art liegt lokal auf der Site, wird aber noch auf keiner Seite angezeigt.
- **Feed-Beschreibung:** Klartext mit Link auf die Folgenseite, dort Quellen, Revisions-Permalinks und Versionsgeschichte. Die Quellen stehen nicht im Feed selbst.
- **Quellen der Folge:** Hauptartikel plus jeder verlinkte Artikel, aus dem mindestens ein verwendeter Fakt stammt (Revisions-ID im Fakt-ID-Präfix). Ein unbekannter Revisions-ID ist ein Fehler, keine stille Auslassung.
- **Keine Folgen- oder Staffelnummern** im Feed (`itunes:episodeType` = full).
- **Explizit:** Feed `false`, pro Folge aus der Prüfung. Auf der Folgenseite steht bei `explicit` ein Hinweis („nicht für alle Hörer geeignet“). Wortlaut von mir.
- **Datum der Folge** auf der Seite ist der Tag des Uploads (`published_at`), nicht der Tag des Laufs.
- **Rechtstexte:** werden aus `site/*.md` zu `impressum.html` und `datenschutz.html` gebaut (Zeilenumbrüche bleiben erhalten). Die Link-Ziele der Erklärseite (`impressum.html`, `datenschutz.html`) stimmen damit.
- **Gestaltung:** eigene, schlichte Seitenvorlage mit der Farbpalette der Erklärseite, ohne Logo. Reine Annahme; Optik ist nicht abgenommen.

## Geänderte Inhalte (bitte gegenlesen)

- **`site/so-funktionierts.html`, Abschnitt 5:** Der Satz „Im Moment liest er jeden dieser Beiträge …“ und der Probezeit-Absatz sind ersetzt. Neu: Die ersten fünf Folgen wurden vollständig geprüft, danach eine Auswahl, die nach fehlerfreien Folgen kleiner und nach einem Fehler größer wird. Das folgt `config/pipeline.toml` (`adjust_rate`, 2/3 und ×2). Nicht geändert: der Satz, dass Andreas auch die Einschätzung zu drastischen Inhalten prüft. Ob das bei Stichproben weiterhin jede Folge betrifft, steht in den Dokumenten nicht; bitte prüfen.
- **`CLAUDE.md`:** Cover-Abschnitt (gebündelte Schrift), auf deine Anweisung.

## Geprüft

- **Erklärseite gegen die Pipeline:** Die Zahlen (20 bis 40 Fakten, 8 Wörter, 10 Fakten, 6 Monate, Stücke von etwa 6 Minuten, gesichtete Fassung) stimmen mit `[quotas]`, D2, D4 und D5 überein. Nicht erwähnt sind die Prüfung auf explizite Inhalte (D7) und Begrüßung/Abspann; das ist keine falsche Aussage, nur eine Lücke.

## Zurückgestellt

- Eine Untergrenze für die Titelschrift im Cover (D7 nennt keine).
- Anzeige des Show-Arts auf der Startseite.
