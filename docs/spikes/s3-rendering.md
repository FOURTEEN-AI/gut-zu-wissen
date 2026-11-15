# Spike S3: Rendering

**Question (plan, Phase 0):** Does `action=parse` per segment differ from in-article rendering? How do infobox parameters map to rendered row labels (D3)?
**Output:** decision between rendering per segment and rendering once and mapping; label-mapping method.
**Result:**
- **Render per segment.** 4,860 of 4,864 sentence segments give exactly the same plain text on their own as in the rendered article. The other 4 are layout artefacts.
- **Infobox labels: marker render.** Each parameter's value is replaced by a marker, and the row that shows the marker gives the label. This mapped every content row of the 8 infoboxes correctly.
- **Change to S2:** an infobox segment is a **rendered row**, not a parameter.

Code: [spikes/s3_rendering.py](../../spikes/s3_rendering.py). Results: [tests/fixtures/wikipedia/spikes/s3_results/](../../tests/fixtures/wikipedia/spikes/s3_results/). Corpus: the 10 S2 revisions. The render cache (`s3_cache/`, 7.8 MB) is not committed; a run re-fetches it in about 1 minute.

## Sentences: per segment vs. in the article

**Method:**
- Every sentence segment is rendered with `action=parse`, `title=<article>`, refs and comments removed first, as the design specifies ("Klartext ohne `<ref>`-Tags").
- Up to 60k characters of segments go into one POST, separated by marker paragraphs (`SEGMENTGRENZE00042`). That's about 50 API calls for 4,864 segments.
- Each revision is rendered once (`action=parse&oldid=…`), tables are removed, and every segment's plain text is searched in the article's plain text.
- Plain text means: HTML text without `sup.reference`, edit links, `style`, `.noprint`, `.metadata`, `.error`. `<br>` becomes a space and whitespace is normalised. NBSP, soft hyphens and zero-width characters are removed.

| Result | Segments |
|---|---:|
| identical to the article text | 4,860 |
| text inside a table in the article (`Interpreter:`, `Compiler:`, list items in a Python table) | 2 |
| layout template at the start of the line (`{{userboxbottom}}` renders `\|}`) | 1 |
| renders empty (`{{GeoQuelle\|DE\|BFN-Karten}}`) | 1 |

Content-bearing templates render the same in both places, for example `{{lang}}`, number formats, `{{CASRN}}` and dates. No template in the corpus depends on position in the article.

**Decision: render per segment.** In production only the offered facts and their context are rendered (about 26 facts plus context), in one batched call per article. Rendering once and mapping back would need an alignment between wikitext and HTML that per-segment rendering avoids.

**Gate for the two artefacts:** a segment whose plain text is empty, or contains `|}` or `{|`, cannot be a fact. This is a cheap addition to the extraction gates in step 4.

## Infobox rows: label mapping

### What doesn't work: matching values

Rendering each parameter value and searching it in the rendered infobox mapped only 97 of 196 parameters, and some of those wrongly:
- `HÖHE` → "Koordinaten", because the height also sits in the hidden geo microformat;
- `KONTRAHENT1` → "Ausgang", because "Normannen" also appears in "Sieg der Normannen";
- `chefname` → the wrong "Vorstand" row.

The causes are short values that appear in several cells, values the template transforms (`ja` → "Sendeturm öffentlich zugänglich", `330` → "330 m"), and multi-column rows.

### Method: marker render

1. Render the unmodified infobox template (`action=parse`, article title as context). This is the baseline.
2. For each parameter, render a copy with that parameter's value replaced by a unique marker word (`Qzmarkerqaaab`). Many copies go into one call.
3. The innermost table row whose **visible** text contains the marker is the parameter's row. Geo microformat and `display:none` spans are removed first. The row's position is its index among all infobox rows.
4. If a word marker shows up nowhere, retry with a digit marker (`7007919`, also searched with thousands separators). That catches parameters the template feeds into calculations, such as `HÖHE`, `DOMINANZ` and `SCHARTENHÖHE`.
5. Group the parameters by row. Each rendered row with a label becomes **one** segment. Its label and value come from the baseline render, so units and formatting are included.

Row types handled:

| Row type | Example | Plain text |
|---|---|---|
| label cell + value cell | `HÖHE`, `HÖHE-BEZUG` → row "Höhe" | `Infobox Berg – Höhe: 2962,06 m ü. NHN` |
| several parameters, one row | `Taxon3_Name`, `Taxon3_WissName` → "Tribus" | `Taxobox – Tribus: Echte Füchse (Vulpini)` |
| label filled from a parameter | `anteilseigner_bezeichnung` + `anteilseigner_name` | `Infobox Fußballunternehmen – Aktionäre: 70,00 %: FC Bayern München e. V. …` |
| comparison row under a header row (battle infobox) | `TRUPPENSTÄRKE1`, `TRUPPENSTÄRKE2` under "Truppenstärke" | `Infobox Militärischer Konflikt – Truppenstärke: Normannen: ca. 7000 Mann; Angelsachsen: ca. 7500 Mann` |

A comparison row is told apart from a row with a parameter label by structure: it sits directly below a single-cell header row. Its column names come from the first comparison row of the table (the parties row).

### Results (8 infoboxes, 196 parameters)

| | Rows | Parameters |
|---|---:|---:|
| Labelled rows, mapped (all checked by hand, all correct) | 97 | 113 |
| Single-cell rows: title, image caption, free notes | 23 | 26 |
| Not shown as text: images, colours, kit patterns, ISO codes, coordinates as numbers, taxon ranks, category parameters | | 54 |
| Marker changes the row structure (Eiffelturm `WEITERES`, Wasser `GHS-Piktogramme`, `EUH`) | | 3 |

Cost: per infobox, one baseline render plus one render per parameter, batched. That's about 3–5 API calls per article, plus a second pass for parameters with calculations.

## Change to S2 (infobox segments)

S2 made one `infobox_row` segment per parameter (`Infobox Berg | HÖHE = 2962.06`). S3 shows that rendered rows and parameters are not one-to-one:
- some rows combine several parameters;
- many parameters render no row;
- some rows need the section header to make sense.

New rule, closer to D3's wording ("Infobox rows"):

- **One segment per rendered labelled row.** The raw text is the lines of its parameters in template order, e.g. `Infobox Berg | HÖHE = 2962.06 | HÖHE-BEZUG = DE-NHN`. The plain text is `Infobox <type> – <label>: <value>` (D3).
- **No segment** for parameters that render no text, for single-cell rows, or where the marker changes the layout.
- **Segment IDs stay `{revid}-{index}`.** Infobox rows are numbered in row order, in the position of the infobox in the article.

This means infobox segmentation in step 3 needs the Wikipedia API (the marker render), not just the wikitext.

## Decisions needed

1. **Section headers in infobox plain text.** Some rows only make sense with their section header. FC Bayern has "Name" three times (Verein, Fußballunternehmen, FC Bayern München II) and "2025/26: 1. Platz" under "Erste Mannschaft". D3 fixes the format „Infobox type – label: value“. Should the section header go in front of the label when there is one: „Infobox Fußballunternehmen – Erste Mannschaft – 2025/26: 1. Platz“?
2. **Single-cell rows with their own label** ("Teil von: Invasion Englands durch Normannen", "Ziele: Wilhelm kämpft für seinen Anspruch …", "P: keine P-Sätze"). Skip them like titles and captions (proposal), or split at the first colon into label and value?
