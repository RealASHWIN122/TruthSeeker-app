import streamlit as st
import os
import yt_dlp
import pandas as pd
import altair as alt
import datetime
import wisper
from dateutil import parser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper


st.set_page_config(
    page_title="Rumor Verifier (Free Edition)",
    page_icon="🕵️",
    layout="wide"
)


with st.sidebar:
    st.header(" Agent Configuration")

    google_api_key = st.text_input("Google Gemini API Key", type="password")
    serpapi_api_key = st.text_input("SerpAPI Key", type="password")

    st.markdown("---")
    st.info(
        "Uses **Google Gemini (Free Tier)** for reasoning and "
        "**Local Whisper** for transcription (no OpenAI cost)."
    )

    st.subheader(" Sources Monitored")
    st.markdown(
        "- Times of India\n"
        "- The Hindu\n"
        "- NDTV\n"
        "- Indian Express\n"
        "- LiveMint\n"
        "- YouTube News Clips"
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Share a rumor or news claim, and I’ll verify it using reputed sources."
        }
    ]


def search_text_articles(query: str, api_key: str):
    """Search reputed Indian news sites using SerpAPI."""
    os.environ["SERPAPI_API_KEY"] = api_key

    sites = [
        "timesofindia.indiatimes.com",
        "thehindu.com",
        "ndtv.com",
        "indianexpress.com",
        "livemint.com"
    ]

    site_filter = " OR ".join([f"site:{s}" for s in sites])
    full_query = f"{query} ({site_filter})"

    search = SerpAPIWrapper()
    results = search.run(full_query)

    articles = []
    for r in results.get("organic_results", []):
        date_text = r.get("date") or ""
        try:
            parsed_date = parser.parse(date_text, fuzzy=True)
        except Exception:
            parsed_date = datetime.datetime.now()

        articles.append({
            "source": r.get("source", "News Article"),
            "title": r.get("title"),
            "link": r.get("link"),
            "snippet": r.get("snippet", ""),
            "date": parsed_date,
            "type": "Article"
        })

    return articles


def search_recent_news_videos(query: str, api_key: str):
    """Search YouTube news videos via SerpAPI."""
    import requests

    params = {
        "engine": "google",
        "q": f"{query} site:youtube.com",
        "tbm": "vid",
        "tbs": "qdr:m",
        "api_key": api_key
    }

    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()

    videos = []
    for v in data.get("video_results", [])[:1]:
        try:
            parsed_date = parser.parse(v.get("date", ""), fuzzy=True)
        except Exception:
            parsed_date = datetime.datetime.now()

        videos.append({
            "source": v.get("source", "YouTube"),
            "title": v.get("title"),
            "link": v.get("link"),
            "snippet": v.get("snippet", ""),
            "date": parsed_date,
            "type": "Video"
        })

    return videos


def transcribe_video_local(video_url: str):
    """Download audio and transcribe using local Whisper."""
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "temp_audio.%(ext)s",
            "quiet": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192"
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        model = wisper.load_model("base")
        result = model.transcribe("temp_audio.mp3")

        os.remove("temp_audio.mp3")
        return result["text"]

    except Exception as e:
        print("Transcription error:", e)
        return ""


def analyze_verification(claim, evidence, transcript, api_key):
    """Analyze evidence using Google Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    evidence_text = ""
    for i, e in enumerate(evidence, start=1):
        evidence_text += (
            f"{i}. [{e['date'].strftime('%Y-%m-%d')}] "
            f"{e['type']} - {e['source']}: {e['snippet']}\n"
        )

    prompt = f"""
You are a professional fact-checking agent.

CLAIM:
"{claim}"

EVIDENCE:
{evidence_text}

VIDEO TRANSCRIPT:
{transcript[:1500]}

TASK:
- Verdict: True / False / Misleading / Unverified
- Earliest source
- Brief timeline explanation

Respond in Markdown.
"""

    return llm.invoke(prompt).content



st.title(" Rumor & News Verifier (Free Edition)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Enter a rumor or news claim..."):

    if not google_api_key or not serpapi_api_key:
        st.error("Please enter both API keys in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status = st.status("🔍 Verifying...", expanded=True)

        status.write(" Searching news articles...")
        articles = search_text_articles(user_input, serpapi_api_key)

        status.write(" Searching YouTube news...")
        videos = search_recent_news_videos(user_input, serpapi_api_key)

        transcript = ""
        if videos:
            status.write(" Transcribing video...")
            transcript = transcribe_video_local(videos[0]["link"])
            videos[0]["snippet"] = transcript[:200]

        evidence = sorted(articles + videos, key=lambda x: x["date"])

        if not evidence:
            result = " No reliable sources found for this claim."
            status.update(label="No Evidence Found", state="error")
        else:
            status.write(" Analyzing with Gemini...")
            result = analyze_verification(user_input, evidence, transcript, google_api_key)
            status.update(label="Verification Complete", state="complete")

        st.markdown(result)

        # Timeline chart
        if evidence:
            st.subheader("Timeline of Reports")
            df = pd.DataFrame([
                {
                    "Date": e["date"],
                    "Source": e["source"],
                    "Type": e["type"],
                    "Title": e["title"]
                }
                for e in evidence
            ])

            chart = alt.Chart(df).mark_circle(size=120).encode(
                x="Date:T",
                y="Source:N",
                color="Type:N",
                tooltip=["Date", "Source", "Title", "Type"]
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": result})
