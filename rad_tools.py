import requests
import xml.etree.ElementTree as ET
from langchain.tools import tool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


retry_policy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)

http = requests.Session()
http.mount("https://", HTTPAdapter(max_retries=retry_policy))


@tool
def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """
    Search PubMed for biomedical research relevant to a query.

    Use this tool when you need to find PubMed articles related
    to a medical, radiography, imaging, radiation, or healthcare
    research question.

    Returns a list of PubMed IDs.
    """

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
    }

    headers = {
        "User-Agent": "RadSearchAI/1.0 (learning project)"
    }

    response = http.get(
        url,
        params=params,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


@tool
def get_article_details(pubmed_ids: list[str]) -> list[dict]:
    """
    Retrieve article details from PubMed using PubMed IDs.

    Use this tool after search_pubmed when you need article titles,
    authors, publication years, journals, and abstracts.

    Returns a list of article dictionaries.
    """

    if not pubmed_ids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "xml",
    }

    headers = {
        "User-Agent": "RadSearchAI/1.0 (learning project)"
    }

    response = http.get(
        url,
        params=params,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    root = ET.fromstring(response.text)

    articles = []

    for pubmed_article in root.findall(".//PubmedArticle"):
        pmid_element = pubmed_article.find(".//PMID")
        title_element = pubmed_article.find(".//ArticleTitle")
        journal_element = pubmed_article.find(".//Journal/Title")
        year_element = pubmed_article.find(".//PubDate/Year")

        abstract_elements = pubmed_article.findall(".//Abstract/AbstractText")

        author_elements = pubmed_article.findall(".//Author")

        pmid = (
            pmid_element.text
            if pmid_element is not None
            else "Unknown"
        )

        title = (
            "".join(title_element.itertext())
            if title_element is not None
            else "No title available"
        )

        journal = (
            journal_element.text
            if journal_element is not None
            else "Unknown journal"
        )

        year = (
            year_element.text
            if year_element is not None
            else "Unknown year"
        )

        abstract = " ".join(
            "".join(element.itertext())
            for element in abstract_elements
        )

        if not abstract:
            abstract = "No abstract available"

        authors = []

        for author in author_elements:
            last_name = author.findtext("LastName")
            fore_name = author.findtext("ForeName")

            if last_name and fore_name:
                authors.append(f"{fore_name} {last_name}")
            elif last_name:
                authors.append(last_name)

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    return articles


if __name__ == "__main__":
    ids = search_pubmed.invoke(
        {
            "query": "occupational radiation exposure radiographers",
            "max_results": 3,
        }
    )

    print("PubMed IDs:")
    print(ids)

    details = get_article_details.invoke(
        {
            "pubmed_ids": ids
        }
    )

    print("\nArticle Details:")

    for article in details:
        print("\n--------------------")
        print("PMID:", article["pmid"])
        print("Title:", article["title"])
        print("Year:", article["year"])
        print("Journal:", article["journal"])
        print("Authors:", ", ".join(article["authors"]))
        print("PubMed:", article["pubmed_url"])
        print("Abstract:", article["abstract"][:500])
