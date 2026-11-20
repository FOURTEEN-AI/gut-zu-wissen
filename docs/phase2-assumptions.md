# Phase 2: Annahmen und offene Fragen

Stand: 2026-11-20. Entstanden, während Andreas abwesend war. Die Quelldokumente (Design, Persona, Plan, Rechtstexte) sind bis auf die genannten Stellen unverändert. Alles hier ist eine Annahme von Claude, bis es bestätigt ist.

## Vor dem Go-live zu tun (nur Andreas)

1. ~~GitHub-Secrets~~ erledigt: `ARCHIVE_S3_ACCESS_KEY` und `ARCHIVE_S3_SECRET_KEY` existieren, dieselben Namen wie in `.env`.
2. ~~Pages aktivieren~~ erledigt (2026-11-20, per API): Quelle „GitHub Actions“, Adresse `https://fourteen-ai.github.io/gut-zu-wissen/` (bestätigt die Annahme zur Adresse).
3. **Erster Upload mit einer Testfolge**, bevor die echte Folge live geht. Der Upload ist nur gegen Fakes getestet (siehe unten).
4. **Spotify-Einreichung** (manuell, einmalig) mit der Feed-URL; die Bestätigungsmail geht an gut-zu-wissen@fourteen-it.de.
5. ~~Datenschutz-Aussagen zu IONOS-Auftragsverarbeitungsvertrag und DPF-Zertifizierung von GitHub bestätigen~~ erledigt (bestätigt von Andreas, 2026-11-20).

## Internet Archive: Testupload vom 2026-11-20 (live)

Mit einer 3-Sekunden-Testdatei, Item `gut-zu-wissen-test_2026-11-20_helgoland` in `test_collection` (Archive-Konto aus `.env`; echtes Archiv und Folgen-Archiv unberührt, Pipeline lief über eine temporäre Archivkopie).

- **Bestätigt:** Die Metadaten-Header beim ersten PUT werden übernommen (Titel, Sprache `ger`, Lizenz CC BY-SA 4.0, Datum, Schlagwörter, Medientyp `audio`). Ein neues Item meldet `item_exists` = falsch. Beide Dateien erscheinen in der Dateiliste (`episode.mp3` 49298 Bytes, `cover.png` 107831 Bytes), danach setzt `publish` die Folge auf `published`.
- **Download-URL:** `archive.org/download/<id>/episode.mp3` antwortet mit 302 und dann 200, `audio/mpeg` mit richtiger Länge; für das Cover `image/png`. Die Weiterleitung bleibt eine Stelle zum Suchen, falls Spotify oder Apple den Feed ablehnen.
- **Dauer bis zur Listung:** einige Minuten (Limit 15 Minuten in der Config; nicht ausgereizt).
- **Abweichung:** Das Archive legt trotz abgeschalteter Derivate eine Vorschaugrafik (`__ia_thumb.jpg`) und eine Torrent-Datei an. Unkritisch.
- **Sammlung:** Zum Testeintrag in `test_collection` hat das Archive automatisch `community` ergänzt, das Item ist öffentlich sichtbar. Das Testitem kann nur über das Archive-Konto gelöscht werden (archive.org, Item verwalten). Ob echte Folgen in `opensource_audio` ohne Prüfung angenommen werden, ist damit **nicht** getestet. Ein Konto ohne Sonderrechte darf nach meiner Kenntnis dort hochladen; bestätigt der erste echte Upload.

## Annahmen zum Internet Archive (weiterhin offen)

- Die Dateiliste eines neuen Items zeigt die Dateien nach höchstens 15 Minuten.
- Item-Kennung: `gut-zu-wissen_<Folgen-ID>`. Bei mehr als 100 Zeichen wird gekürzt und ein Hash angehängt.

## Annahmen zur Site und zum Feed

- **Adresse:** `https://fourteen-ai.github.io/gut-zu-wissen` (aus der User-Agent-Zeichenkette in `github.py` abgeleitet). Bei eigener Domain `[site] base_url` in der Config ändern; die Feed-URL ändert sich dann, und Spotify müsste neu eingereicht werden.
- **Folgentitel im Feed und auf der Seite:** der Artikeltitel (kein „Gut zu wissen:“-Präfix). Im MP3 und im Archive-Item steht dagegen „Gut zu wissen: <Titel>“.
- **Cover:** Der Site-Build rendert es pro Folge aus Titel, Template und gebündelter Schrift (`covers/<Folgen-ID>.png`, 3000 px) und liefert es von Pages selbst aus. Es steht auf der Folgenseite und ist die `itunes:image` der Folge (direkte, stabile URL statt der weiterleitenden Archive-URL). Das Archive-Item bekommt es zusätzlich. Folge: Ändert sich das Template, ändern sich auch die alten Cover.
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
