"""Spike helper: fetch the latest sighted revision of sample articles (D2) as fixtures.

Writes one JSON per article to tests/fixtures/wikipedia/spikes/ (CC BY-SA, see NOTICE).
Usage: uv run --group spikes python spikes/fetch_articles.py
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

API = "https://de.wikipedia.org/w/api.php"
USER_AGENT = "GutZuWissen/0.1 (spike; gut-zu-wissen@fourteen-it.de)"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "wikipedia" / "spikes"

# Chosen for structural variety: infoboxes, taxobox, tables, lists, math, code,
# nested templates, galleries, many references.
ARTICLES = [
    "Rotfuchs",  # Taxobox
    "Wasser",  # Infobox Chemikalie, formulas
    "Mount Everest",  # Infobox Berg, coordinates
    "FC Bayern München",  # tables, lists, Infobox Fußballklub
    "Satz des Pythagoras",  # <math>
    "Python (Programmiersprache)",  # <syntaxhighlight>
    "Johann Sebastian Bach",  # long lists of works, dead person
    "Eiffelturm",  # Infobox Bauwerk, numbers with units
    "Zugspitze",  # tables, nested templates
    "Schlacht bei Hastings",  # history, many refs
]


def get(client: httpx.Client, **params: str) -> dict:
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    response = client.get(API, params=params)
    response.raise_for_status()
    return response.json()


def fetch(client: httpx.Client, title: str) -> dict:
    info = get(client, action="query", prop="info|flagged", titles=title, redirects="1")
    page = info["query"]["pages"][0]
    flagged = page.get("flagged")
    if flagged and "stable_revid" in flagged:
        revid, status = flagged["stable_revid"], "sighted"
    else:
        revid, status = page["lastrevid"], "latest"
    revs = get(
        client,
        action="query",
        prop="revisions|categories",
        revids=str(revid),
        rvprop="ids|timestamp|content",
        rvslots="main",
        cllimit="max",
    )
    rpage = revs["query"]["pages"][0]
    rev = rpage["revisions"][0]
    return {
        "title": rpage["title"],
        "revid": rev["revid"],
        "revision_status": status,
        "revision_timestamp": rev["timestamp"],
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "categories": [c["title"] for c in rpage.get("categories", [])],
        "wikitext": rev["slots"]["main"]["content"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for title in ARTICLES:
            article = fetch(client, title)
            name = article["title"].replace(" ", "_").replace("/", "_")
            path = OUT / f"{name}.json"
            path.write_text(json.dumps(article, ensure_ascii=False, indent=2) + "\n")
            print(f"{article['title']}: revid {article['revid']} ({article['revision_status']})")
            time.sleep(1)


if __name__ == "__main__":
    main()
