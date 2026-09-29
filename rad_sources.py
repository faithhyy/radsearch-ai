import json


def extract_pubmed_sources(result: dict) -> list[dict]:
    """Return unique PubMed articles retrieved during an agent run."""

    sources = []
    seen_pmids = set()

    for message in result.get("messages", []):
        if getattr(message, "name", None) != "get_article_details":
            continue

        content = message.content

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue

        if not isinstance(content, list):
            continue

        for article in content:
            if not isinstance(article, dict):
                continue

            pmid = str(article.get("pmid", "")).strip()

            if not pmid or pmid == "Unknown" or pmid in seen_pmids:
                continue

            seen_pmids.add(pmid)
            sources.append(
                {
                    "pmid": pmid,
                    "title": article.get("title", "Untitled article"),
                    "authors": article.get("authors", []),
                    "journal": article.get("journal", "Unknown journal"),
                    "year": article.get("year", "Unknown year"),
                    "abstract": article.get(
                        "abstract", "No abstract available"
                    ),
                    "url": article.get("pubmed_url")
                    or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

    return sources
