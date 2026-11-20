# S8: Artikel des Tages (2026-11-20)

**Frage.** Woher kommt der Artikel des Tages der deutschsprachigen Wikipedia verlässlich, und woher
kommen die Ersatzartikel (Exzellent/Lesenswert)?

**Messung** (live gegen die MediaWiki-API, 2026-11-20).

- Ein `featured`-Feed existiert nicht (Annahme des Plans bestätigt).
- `Wikipedia:Hauptseite/Artikel des Tages/Chronologie 2026` listet pro Tag eine Zeile
  `* 20.11.2026 [[Volksmudschahedin]]`. Der Bot TaxonBot schreibt sie jeden Tag um **23:00 UTC**
  für den folgenden Tag (Zusammenfassung „heutiger Artikel des Tages"), zuletzt zwölf Tage in Folge
  gemessen. Um 03:17 UTC steht der Artikel des Tages also seit gut vier Stunden dort, unabhängig
  davon, ob man nach UTC oder deutscher Zeit rechnet.
- Die Wochentagsseiten (`…/Freitag`) tragen `DATUM` und `LEMMA` mit demselben Titel; sie werden von
  Hand gepflegt und dienen als Rückfall (`DATUM` muss das gesuchte Datum sein).
- „(erneut)" hinter einem Titel kommt vor; der Link bleibt der erste in der Zeile.
- Ersatzartikel: `Kategorie:Wikipedia:Exzellent` (3 021 Artikel) und
  `Kategorie:Wikipedia:Lesenswert` (4 486). Mit `cmlimit=500` sind es 15 Anfragen, bei einer Anfrage
  pro Sekunde etwa 15 Sekunden, nur wenn ein Ersatz gebraucht wird.

**Entscheidung** (Annahme, im Plan nicht festgelegt): Chronologie zuerst, Wochentagsseite als
Rückfall, sonst Abbruch mit Hinweis. Ersatz zufällig, gesät mit dem Datum (nachvollziehbar), Gates
vor der Auswahl.

**Nicht gemessen.** Wie oft der Artikel des Tages an einem Tag nachträglich geändert wird.
