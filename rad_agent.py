import os
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from rad_tools import search_pubmed, get_article_details


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY was not found. Check your .env file."
    )


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


tools = [
    search_pubmed,
    get_article_details,
]


SYSTEM_PROMPT = """
You are RadSearch AI, a biomedical literature research assistant
focused on radiography, medical imaging, radiation safety,
radiotherapy, and related healthcare research.

Your job is to answer biomedical research questions using
evidence retrieved from PubMed.

Rules:

1. Use the PubMed tools before making research claims.

2. Use search_pubmed to discover relevant studies.

3. Use get_article_details to retrieve the title, authors,
   journal, year, abstract, and PMID of relevant studies.

4. Only discuss studies that were actually returned by the tools.

5. Never invent article titles, authors, journals, publication
   years, PMIDs, or study findings.

6. Cite each study in the answer using its PMID.

7. If the available evidence is weak, conflicting, limited,
   or insufficient, say so clearly.

8. Distinguish evidence from interpretation.

9. Do not diagnose patients.

10. Do not provide patient-specific treatment recommendations.

11. Do not claim that research proves something unless the
    retrieved evidence genuinely supports that level of certainty.

12. If the question is outside biomedical research, explain that
    RadSearch AI is designed for biomedical literature research.

Produce a clear, naturally structured evidence summary based only on
retrieved PubMed information. Use concise paragraphs and lists where they
improve readability. The Streamlit interface displays the retrieved PubMed
articles separately, so do not repeat a Sources section or article links in
the written answer.
"""


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


def run_radsearch(question: str):
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    return result
