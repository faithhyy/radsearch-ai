import sys
import os
import re

sys.path.insert(0, ".")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from rad_tools import search_pubmed, get_article_details


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

api_key_loaded = bool(os.getenv("OPENAI_API_KEY"))

if not api_key_loaded:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Check that your .env file contains OPENAI_API_KEY=your_key"
    )


# --------------------------------------------------
# 2. INITIALIZE THE LLM
# --------------------------------------------------

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# --------------------------------------------------
# 3. DEFINE THE TOOLS AVAILABLE TO THE AGENT
# --------------------------------------------------

tools = [
    search_pubmed,
    get_article_details,
]


# --------------------------------------------------
# 4. CREATE THE AGENT
# --------------------------------------------------

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
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

6. Include the PMID for every study you cite.

7. If the available evidence is weak, conflicting, limited,
   or insufficient, say so clearly.

8. Distinguish evidence from interpretation.

9. Do not diagnose patients.

10. Do not provide patient-specific treatment recommendations.

11. Do not claim that research proves something unless the
    retrieved evidence genuinely supports that level of certainty.

12. If the question is outside biomedical research, explain that
    RadSearch AI is designed for biomedical literature research.

Produce a clear evidence summary based only on retrieved PubMed
information.
"""
)


# --------------------------------------------------
# 5. TEST QUESTION
# --------------------------------------------------

question = """
What does research say about occupational radiation exposure
among radiographers?
"""


# --------------------------------------------------
# 6. RUN THE AGENT
# --------------------------------------------------

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


# --------------------------------------------------
# 7. GET THE FINAL ANSWER
# --------------------------------------------------

final_answer = result["messages"][-1].content


# --------------------------------------------------
# 8. COLLECT VALID PMIDs FROM TOOL RESULTS
# --------------------------------------------------

valid_pmids = set()

for message in result["messages"]:
    if type(message).__name__ == "ToolMessage":
        content = str(message.content)

        found_pmids = re.findall(
            r"\b\d{7,8}\b",
            content
        )

        for pmid in found_pmids:
            valid_pmids.add(pmid)


# --------------------------------------------------
# 9. FIND PMIDs USED IN THE FINAL ANSWER
# --------------------------------------------------

answer_pmids = set(
    re.findall(
        r"\b\d{7,8}\b",
        final_answer
    )
)


# --------------------------------------------------
# 10. VERIFY THE CITATIONS
# --------------------------------------------------

invalid_pmids = answer_pmids - valid_pmids

if invalid_pmids:
    print("\n⚠️ PMID VERIFICATION WARNING")

    print(
        "The final answer mentioned PMID(s) "
        "that were not found in the retrieved tool results:"
    )

    print(invalid_pmids)

else:
    print("\n✅ PMID VERIFICATION PASSED")


# --------------------------------------------------
# 11. PRINT A CLEAN AGENT TRACE
# --------------------------------------------------

print("\n=== AGENT TRACE ===")

for message in result["messages"]:

    message_type = type(message).__name__

    if message_type == "HumanMessage":

        print("\nUSER:")
        print(message.content)

    elif message_type == "AIMessage":

        if getattr(message, "tool_calls", None):

            for tool_call in message.tool_calls:

                print("\nAGENT CALLED TOOL:")
                print("Tool:", tool_call["name"])
                print("Arguments:", tool_call["args"])

        elif message.content:

            print("\nFINAL ANSWER:")
            print(message.content)

    elif message_type == "ToolMessage":

        print("\nTOOL RESULT:")

        tool_content = str(message.content)

        print(tool_content[:1500])


# --------------------------------------------------
# 12. PRINT FINAL ANSWER SEPARATELY
# --------------------------------------------------

print("\n\n=== RADSEARCH AI FINAL RESPONSE ===")

print(final_answer)