import os
import yt_dlp
import datetime
import whisper
import requests
from dateutil import parser
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper

app = FastAPI(
    title="Rumor Verifier API",
    description="API for verifying news claims using Gemini and SerpAPI",
    version="1.0.0"
)



class VerificationRequest(BaseModel):
    claim: str
    google_api_key: str
    serpapi_api_key: str

class EvidenceItem(BaseModel):
    source: str
    title: Optional[str] = None
    link: Optional[str] = None
    snippet: str
    date: datetime.datetime
    type: str

class VerificationResponse(BaseModel):
    verdict_text: str
    evidence_timeline: List[EvidenceItem]



def search_text_articles(query: str, api_key: str) -> List[dict]:
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
   
    if isinstance(results, dict):
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

def search_recent_news_videos(query: str, api_key: str) -> List[dict]:
    """Search YouTube news videos via SerpAPI."""
    params = {
        "engine": "google",
        "q": f"{query} site:youtube.com",
        "tbm": "vid",
        "tbs": "qdr:m",python
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

def transcribe_video_local(video_url: str) -> str:
    """Download audio and transcribe using local Whisper."""
    temp_filename = "temp_audio"
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{temp_filename}.%(ext)s",
            "quiet": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192"
            }]
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        model = whisper.load_model("base")
        result = model.transcribe(f"{temp_filename}.mp3")

        if os.path.exists(f"{temp_filename}.mp3"):
            os.remove(f"{temp_filename}.mp3")
            
        return result["text"]
    except Exception as e:
        print("Transcription error:", e)
       
        if os.path.exists(f"{temp_filename}.mp3"):
            os.remove(f"{temp_filename}.mp3")
        return ""

def analyze_verification(claim: str, evidence: List[dict], transcript: str, api_key: str) -> str:
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



@app.post("/verify", response_model=VerificationResponse)
def verify_claim_endpoint(request: VerificationRequest):
    """
    Accepts a claim and API keys, searches for evidence, and returns a verdict.
    """
    try:
        articles = search_text_articles(request.claim, request.serpapi_api_key)
        videos = search_recent_news_videos(request.claim, request.serpapi_api_key)

        transcript = ""
        if videos:
            transcript = transcribe_video_local(videos[0]["link"])
            videos[0]["snippet"] = transcript[:200]

        evidence = sorted(articles + videos, key=lambda x: x["date"])

        if not evidence:
            return VerificationResponse(
                verdict_text="No reliable sources found for this claim.",
                evidence_timeline=[]
            )

        result = analyze_verification(request.claim, evidence, transcript, request.google_api_key)

        return VerificationResponse(
            verdict_text=result,
            evidence_timeline=evidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))