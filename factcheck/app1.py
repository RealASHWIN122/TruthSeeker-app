from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from duckduckgo_search import DDGS
import time

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

app = FastAPI(title="TruthSeeker Local Fact-Check API")

print("Waking up the local AI... (This might take 10-20 seconds)")
try:
    # Load the Phi-3 model into your laptop's RAM
    llm = Llama(
        model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=4096,      # Memory size for reading search results
        n_gpu_layers=0,  # 0 = Run on CPU (perfect for integrated graphics)
        verbose=False    # Keeps the terminal output clean
    )
    print("✅ AI is awake and ready!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'Phi-3-mini-4k-instruct-q4.gguf' is inside a folder named 'models'!")

# Define what the Android App will send us
class CheckRequest(BaseModel):
    claim_text: str

def web_search(query: str):
    """Searches DuckDuckGo silently in the background."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No recent news found."
        
        # Format the search results cleanly
        evidence = "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in results])
        return evidence
    except Exception as e:
        print(f"Search error: {e}")
        return "Search unavailable."

# The actual API Endpoint the Android App hits
@app.post("/verify")
def verify_claim(request: CheckRequest):
    claim = request.claim_text
    
    print(f"\n🔍 Fact-checking: {claim}")
    
    # 1. Grab live data from the web
    evidence = web_search(claim)
    
    # 2. Build the exact prompt structure that Phi-3 expects
    prompt = f"""<|user|>
You are an expert Fact-Checking AI. Verify this claim using ONLY the provided evidence.

CLAIM: "{claim}"

EVIDENCE FROM WEB:
{evidence}

TASK:
1. Verdict: Is it True, False, Misleading, or Unverified?
2. Explanation: Explain why in 1 or 2 short sentences.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
Verdict: [Your Verdict]
Explanation: [Your Explanation]<|end|>
<|assistant|>"""

    # 3. Make the AI think and generate the answer
    start_time = time.time()
    output = llm(
        prompt, 
        max_tokens=150, 
        temperature=0.1,  # Low temperature = strict, factual answers
        stop=["<|end|>"], 
        echo=False
    )
    
    result_text = output['choices'][0]['text'].strip()
    print(f"⚡ Done in {round(time.time() - start_time, 2)} seconds.")
    
    # Send the JSON back to the Android phone
    return {
        "result": result_text, 
        "evidence_used": evidence
    }

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" is magic: it allows your phone to connect over your WiFi network!
    uvicorn.run(app, host="0.0.0.0", port=8000)