# Phase 3: Annahmen und offene Punkte

Stand: 2026-11-20. Gebaut nach deiner Vorgabe „Baue Phase 3; Veröffentlichung verzögern, damit ich eingreifen kann, sonst automatisch veröffentlichen". Alles hier ist eine Annahme, die du ändern kannst.

## Annahmen

1. **Feste Uhrzeit 10:00 UTC** (11 Uhr deutscher Winterzeit, 12 Uhr im Sommer), geändert nach deiner Entscheidung vom 2026-11-20. Die Folge muss mindestens 2 Stunden vorher fertig sein (`[publishing] min_age_hours`); der Tageslauf endet meist gegen 5 bis 6 Uhr deutscher Zeit. Eine später fertige Folge wartet bis zum nächsten Tag. Uhrzeit: Cron in `auto-publish.yml`.
2. **Bremse = Label `hold` am Review-Issue.** Ohne Label läuft die Veröffentlichung. Mit Label wartet die Folge, bis du es entfernst (oder von Hand veröffentlichst). Das Label muss im Repo einmal angelegt werden; GitHub legt es beim ersten Setzen an.
3. **Review-Fehler bremsen automatisch.** Ein Review mit `outcome: error` verhindert die automatische Veröffentlichung. Ist zur Veröffentlichungszeit noch kein Review da (Stichprobe nicht beantwortet), wird trotzdem veröffentlicht, wie bei der Antikythera-Folge.
4. **Veröffentlichung auch bei `explicit: true`.** Das Feed-Flag wird gesetzt, so wie es D7 vorsieht.
5. **Benachrichtigung = fehlgeschlagener Workflow-Lauf.** Ein verworfener Tageslauf (Ersatzartikel erschöpft, Jev-Abbruch) oder ein Absturz beendet den Lauf mit Fehler; GitHub schickt die Standard-Mail. Es gibt keine eigene Benachrichtigung.
6. **Ohne Artikel des Tages (Chronologie und Wochentagsseite ohne Eintrag)** nimmt der Lauf den ersten erlaubten Ersatzartikel statt abzubrechen.
7. **Ein Lauf pro Tag**, höchstens eine Folge pro Lauf. Bei Rückstau folgt die nächste am nächsten Tag oder per manuellem Start des Workflows.
8. **Ein Absturz** (nicht `GateFailure`) im Tageslauf sucht keinen Ersatzartikel, sondern eskaliert mit Traceback, so wie im manuellen Lauf.

## Offen für dich

- **Erklärseite (`site/so-funktionierts.html`):** Sie sagt, dass Andreas die Beiträge am Ende liest. Mit der automatischen Veröffentlichung kann eine Folge vor dem Review erscheinen. Ich habe den Text nicht geändert, das entscheidest du.
- **Kosten:** Etwa 1,5 bis 2,5 USD je Folge, also grob 45 bis 75 USD im Monat vor Fehlversuchen. Ein Ersatzartikel verdoppelt den Tag.
- **Erster Live-Lauf:** Der Tageslauf wurde nur mit Fakes getestet (`tests/test_daily.py`) und die Wikipedia-Abfragen einmal live gelesen. Der erste echte Lauf (Cron 03:17 UTC oder `workflow_dispatch`) ist der Test. Beachte: Er veröffentlicht nach 6 Stunden von allein.
- **Nicht getestet:** `auto-publish.yml` (Workflow-Syntax, Artifact-Download über Läufe hinweg, Dispatch der Pages-Workflow).

## Später ergänzt (autonome Arbeit, 2026-11-20)

9. **Cron bleibt an.** Auf deine Anweisung „arbeite eigenständig weiter" habe ich den Tageslauf nicht deaktiviert. Der erste Lauf kommt um 03:17 UTC (4:17 Uhr deutscher Zeit am 21.11.) und wird um 10:00 UTC (11 Uhr deutscher Zeit) von allein veröffentlicht, wenn niemand `hold` setzt.
10. **Label `hold` ist angelegt** (rot, mit Beschreibung), damit du es am Issue nur noch auswählen musst.
11. **Pages in zwei Jobs getrennt** (Build, Deploy). Der gemeinsame Job scheiterte in drei von vier Läufen mit „No artifacts named github-pages". Der Testlauf nach der Änderung war grün; ob das Problem damit endgültig weg ist, zeigen die nächsten Läufe.
12. **README:** Secret-Namen korrigiert (`ARCHIVE_S3_*`, vorher fälschlich `IA_S3_*`), Abschnitt zu Veröffentlichung und Tageslauf ergänzt.
