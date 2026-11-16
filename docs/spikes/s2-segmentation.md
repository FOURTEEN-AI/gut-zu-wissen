# Spike S2: SoMaJo + masking

**Question (plan, Phase 0):** Is sentence segmentation of raw wikitext robust on diverse articles (nested templates, tables, lists)?
**Output:** masking rules, segment ID scheme.
**Result:** Go. The masking rules below give exact round trips and deterministic output on all 10 articles. Decisions are recorded at the end.

Code: [spikes/s2_segmentation.py](../../spikes/s2_segmentation.py). Corpus: 10 latest sighted revisions in [tests/fixtures/wikipedia/spikes/](../../tests/fixtures/wikipedia/spikes/), fetched with [spikes/fetch_articles.py](../../spikes/fetch_articles.py). The articles were chosen for structure: Taxobox, chemistry infobox, tables, `<math>`, `<syntaxhighlight>`, long lists, many refs.

## Results

| Article | Sentence segments | Infobox rows |
|---|---:|---:|
| Eiffelturm | 760 | 30 |
| FC Bayern München | 1222 | 58 |
| Johann Sebastian Bach | 589 | 0 |
| Mount Everest | 554 | 18 |
| Python (Programmiersprache) | 210 | 7 |
| Rotfuchs | 367 | 20 |
| Satz des Pythagoras | 254 | 0 |
| Schlacht bei Hastings | 104 | 17 |
| Wasser | 345 | 25 |
| Zugspitze | 459 | 21 |
| **Total** | **4864** | **196** |

Invariants checked on every article:

- **Exact round trip:** for every prose unit, the segments plus the whitespace between them reproduce the raw text byte for byte.
- **Determinism:** two runs give identical segments.
- **Markup:** no segment has unbalanced `[[…]]`, `{{…}}` or `<ref>…</ref>`.
- **No dropped refs:** every `<ref>` in prose lands in a segment. The 190 refs outside segments are in excluded sections (85), tables (65), templates incl. infoboxes (33), file links (4), and after display math or inside a quote template (3).

Runtime: about 0.6 s per article.

### Heuristic flags (sentence segments)

| Flag | Count | What it is |
|---|---:|---|
| no end punctuation | 267 | Mostly list items (filmographies, weblink lists, "named after" lists) and display-math lines. None is a sentence torn in two. |
| lower-case start | 37 | List items continuing an intro ("Wasser hat" + "eine geringe Wärmeleitfähigkeit …") and text after display math ("wobei …", "gilt."). |
| < 4 words | 91 | List items, years in filmographies, intros like "Daraus folgt". |
| > 60 words | 27 | Genuinely long sentences; checked by hand, no missed splits. |
| unbalanced markup | 0 | |

All flagged segments fail the step 4 gates (≥ 8 words, full sentence), so they never become facts. A search for missed splits ("word. Capital") found 13 hits. 6 were real and are fixed by the false-abbreviation rule below; the other 7 are Roman numerals or punctuation inside link labels and titles ("XXXIII. Olympische", "Sehet! Wir gehn …").

## Masking rules

Raw wikitext is scanned once. Each construct becomes a mask with a *surrogate*: the text SoMaJo sees in its place. Segment boundaries are mapped back to raw offsets. A segment's raw text always includes the full markup.

| Construct | Surrogate | Why |
|---|---|---|
| `[[target\|label]]`, `[[target]]` (nested allowed) | visible label | SoMaJo sees real words ("J. S. Bach" is handled as abbreviations). **No split inside a link**; a split there is merged back. |
| `[[Datei:…]]`, `[[Bild:…]]`, `[[Kategorie:…]]` | nothing | Not prose. |
| `[http://… label]` | label | as for links |
| `{{…}}` (balanced, nested) | the word `Vorlage` | Behaves like a word mid-sentence; no sentence ends inside a template. |
| `{\| … \|}` at line start (nested) | `Vorlage` | Tables are blocks, not prose. |
| `<ref …>…</ref>`, `<ref …/>`, `<!-- … -->` | nothing; **attaches to the preceding sentence** | Refs follow the punctuation in dewiki. |
| `''`, `'''` (bold/italic) | nothing; **attaches to the following sentence** | Otherwise `'''Wasser''' ist …` hides the sentence start from SoMaJo. |
| `<math>`, `<chem>`, `<ce>`, `<code>` inline | `Vorlage` | Formula is part of the sentence. |
| other extension tags (`gallery`, `syntaxhighlight`, `pre`, `poem`, `timeline`, `score`, `references`, …) | nothing | Not prose. |
| HTML entities (`&nbsp;`, `&ndash;`, …) | decoded character | SoMaJo otherwise splits `&nbsp;` into three tokens. |
| `°C`, `°F` | `Grad` | SoMaJo reads "°C." as the abbreviation "C." and never splits after it. |
| scores and times `3:1`, `21:30` | `Zahl` | SoMaJo reads "3:1." as the ordinal "1." and never splits after it. |
| At the end of a unit | everything attaches left | Trailing `''` and refs after the final full stop stay in the last sentence. |

**Units.** The masked text is split into lines first:

- A heading starts a new section.
- A line with only non-prose masks (block template, table, display math, file) is a *block*. Indented (`:`) and list lines count as blocks when only such masks remain after the list marker.
- Each list item (`*`, `#`, `:`, `;`) is its own unit.
- Consecutive plain lines form one paragraph unit; a single newline is a soft break.
- Lines starting with a space (preformatted text), `----` and `__MAGIC__` are skipped.

**Hatnotes:** a line starting with one of the templates *Dieser Artikel, Hauptartikel, Siehe auch, Begriffsklärungshinweis, Weiterleitungshinweis, Andere Bedeutung, Unterscheide* is skipped as a whole. This includes text after the template on the same line, which renders as part of the note (Wasser: `{{Hauptartikel|…}} mit allen …`). The list is extended as cases are found.

**Excluded sections:** level-2 sections *Literatur, Weblinks, Einzelnachweise, Anmerkungen, Siehe auch, Quellen, Belege, Fußnoten*, including their subsections.

**False abbreviations.** SoMaJo's German abbreviation list matches case-insensitively and contains entries that are ordinary words in prose. It never ends a sentence after them. The corpus showed three: `Italien.`, `Japan.` and `mag.` (from "Mag." = Magister). After SoMaJo, a sentence is split after such a token when the next token is capitalised. This was found by checking which list entries occur in the corpus both with and without a full stop.

**Infobox rows.** *Superseded by S3:* one segment per rendered labelled row, with section header and colon-split single-cell rows, mapped with the marker render; see [s3-rendering.md](s3-rendering.md#change-to-s2-infobox-segments). Original S2 rule: a block template named `Infobox …` or `Taxobox` (D3) is split on top-level `|`. Each named parameter with a non-empty value (comments removed) becomes one `infobox_row` segment. The raw text is `<template name> | <param> = <value>`. Positional parameters are skipped. `Personendaten` is excluded because only `Infobox *` and `Taxobox` match.

## Segment ID scheme

`{revid}-{index:04d}`, for example `271590031-0042`. The index counts sentence and infobox-row segments together, in document order from 0.

- Stable for a given revid and segmenter: the output is deterministic.
- Not stable across segmenter changes: a rule change shifts indices. The archive stores the segments themselves, so IDs inside one episode stay consistent. Re-segmenting an old revid with new code can produce different IDs.
- Four digits cover the largest article in the corpus (FC Bayern, 1,280 segments). Longer articles get a fifth digit; the format does not need a fixed width.

## Known limits

- **List content.** List items are segmented but are rarely full sentences, so they fail the full-sentence gate. Facts that only exist in lists are lost.
- **Sentences around display math.** A sentence with a formula on its own line ("…, dass die Gleichung / : <math>…</math> / gilt.") is split into fragments, which the gate drops.
- **Non-content infobox rows.** Rows like `rightarm1 = FF0000` (shirt colours) or image file names are kept as segments. Extraction can ignore them; mapping parameters to rendered labels is S3.

## Decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Excluded sections | The list above as it stands; *Filmografie*, *Diskografie*, *Werke* stay in (the gates drop their list items). |
| 2 | List content | Accepted: facts stated only in list items are lost. |
| 3 | Segmenter version | `{revid}-{index}` as in the plan; no segmenter version in the archive for now. Added once there is a second segmenter version. |
| 4 | False abbreviations | Fixed list (`Italien.`, `Japan.`, `mag.`), extended as cases are found. |
| 5 | Hatnotes with trailing text | A line starting with a hatnote template (fixed list, see masking rules) is skipped as a whole; the list is extended as cases are found. |
| 6 | Dependency freshness | No pin; resolved versions stay as they are. |

The spike dependency group (SoMaJo, httpx) moves into the main dependencies in step 3.
