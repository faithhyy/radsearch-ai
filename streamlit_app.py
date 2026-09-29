import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
from PIL import Image

sys.path.insert(0, ".")

from rad_agent import run_radsearch
from rad_sources import extract_pubmed_sources


logo_path = Path(__file__).parent / "assets" / "radsearch-logo.png"
logo = Image.open(logo_path)

st.set_page_config(
    page_title="RadSearch AI",
    page_icon=logo,
    layout="centered",
)


st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFFFF;
        }
        h1, h2, h3 {
            color: #205493;
        }
        p, label, .stMarkdown, textarea, input {
            color: #111827;
        }
        textarea, input {
            background-color: #F1F1F1 !important;
            color: #212121 !important;
            border-color: #AEB0B5 !important;
        }
        textarea::placeholder, input::placeholder {
            color: #5B616B !important;
        }
        a {
            color: #0071BC !important;
        }
        .stButton > button[kind="primary"] {
            background-color: #205493;
            border-color: #205493;
            color: #FFFFFF;
            min-height: 44px;
            font-weight: 600;
            width: 100%;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #112E51;
            border-color: #112E51;
        }
        [data-testid="stSidebar"] {
            background-color: #F1F1F1;
            border-right: 1px solid #D6D7D9;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #D6D7D9;
            background-color: #FFFFFF;
        }
        [data-testid="stLinkButton"] a {
            color: #0071BC !important;
            border-color: #0071BC !important;
            background-color: #FFFFFF !important;
            font-weight: 600;
        }
        [data-testid="stLinkButton"] a:hover {
            color: #FFFFFF !important;
            background-color: #0071BC !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


logo_column, title_column = st.columns([1, 5], vertical_alignment="center")

with logo_column:
    st.image(logo, width=88)

with title_column:
    st.title("RadSearch AI")
    st.caption(
        "Biomedical literature assistant for radiography and medical imaging"
    )


if "history" not in st.session_state:
    st.session_state.history = []

if "active_result" not in st.session_state:
    st.session_state.active_result = None

if "last_search" not in st.session_state:
    st.session_state.last_search = {"question": "", "started_at": 0.0}


with st.sidebar:
    st.subheader("Prior Sessions")

    if st.session_state.history:
        for index, session in enumerate(reversed(st.session_state.history)):
            if st.button(
                session["question"],
                key=f"history_{index}",
                use_container_width=True,
            ):
                st.session_state.active_result = session

            st.caption(session["time"])

        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.active_result = None
            st.rerun()
    else:
        st.caption("Your previous research questions will appear here.")


question = st.text_area(
    "Ask a research question",
    placeholder="Example: What does research say about radiation dose reduction in pediatric CT?",
    height=120,
    max_chars=1000,
)


search_button = st.button(
    "Search PubMed",
    type="primary",
    icon=":material/search:",
    use_container_width=True,
)


if search_button:
    clean_question = question.strip()
    normalized_question = " ".join(clean_question.lower().split())
    last_search = st.session_state.last_search
    repeated_too_soon = (
        normalized_question == last_search["question"]
        and time.monotonic() - last_search["started_at"] < 60
    )

    if not clean_question:
        st.warning("Please enter a research question first.")
    elif repeated_too_soon:
        st.info(
            "This question was just searched. Please review the current result "
            "or wait a moment before searching it again."
        )
    else:
        try:
            with st.spinner("Searching PubMed and reviewing evidence..."):
                result = run_radsearch(clean_question)
                final_answer = result["messages"][-1].content
                sources = extract_pubmed_sources(result)

            session = {
                "question": clean_question,
                "answer": final_answer,
                "sources": sources,
                "time": datetime.now().strftime("%d %b %Y, %H:%M"),
            }
            st.session_state.history.append(session)
            st.session_state.active_result = session
            st.session_state.last_search = {
                "question": normalized_question,
                "started_at": time.monotonic(),
            }
        except requests.exceptions.HTTPError as error:
            if error.response is not None and error.response.status_code == 429:
                st.error(
                    "PubMed is receiving too many requests right now. "
                    "Please wait a moment and try again."
                )
            else:
                st.error(
                    "PubMed is temporarily unavailable. Please try again shortly."
                )
        except requests.exceptions.RequestException:
            st.error(
                "RadSearch could not reach PubMed. Check your connection and try again."
            )
        except RateLimitError:
            st.error(
                "The AI service is receiving too many requests right now. "
                "Please wait a moment and try again."
            )
        except AuthenticationError:
            st.error(
                "RadSearch is temporarily unavailable because its AI service "
                "could not be verified."
            )
        except APIConnectionError:
            st.error(
                "RadSearch could not reach the AI service. Check your connection "
                "and try again."
            )
        except APIError:
            st.error(
                "The AI service is temporarily unavailable. Please try again shortly."
            )
        except Exception:
            st.error(
                "RadSearch could not complete this search. Please try again shortly."
            )


active_result = st.session_state.active_result

if active_result:
    st.divider()
    st.subheader("Overall Summary")

    with st.container(border=True):
        st.markdown(active_result["answer"])

    st.subheader("Retrieved PubMed Articles")

    sources = active_result["sources"]

    if sources:
        for source in sources:
            with st.container(border=True):
                st.markdown(f"**{source['title']}**")
                authors = ", ".join(source.get("authors", []))

                if authors:
                    st.markdown(f"**Authors:** {authors}")

                st.markdown(
                    f"{source['journal']} ({source['year']}) | "
                    f"PMID: {source['pmid']}"
                )
                abstract = source.get("abstract")

                if abstract:
                    st.markdown(f"**Abstract:** {abstract}")
                st.link_button(
                    "View on PubMed",
                    source["url"],
                    icon=":material/open_in_new:",
                )
    else:
        st.info("No PubMed articles were returned for this question.")


st.divider()
st.caption(
    "RadSearch AI supports literature research and does not provide medical "
    "diagnosis or patient-specific treatment advice. Always review the "
    "original publications."
)
