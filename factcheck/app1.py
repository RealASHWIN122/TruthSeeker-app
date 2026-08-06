from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from ddgs import DDGS
import time

app = FastAPI(title="Skyra Phi-3 Fact-Checker")

# --- 1. LOAD THE LLM ---
print("Waking up Phi-3... (Running on CPU for stability)")
try:
    # Load the Phi-3 model
    llm = Llama(
        model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=2048,      # Context window for search results
        n_gpu_layers=0,  # 0 = CPU only (prevents VRAM crashes)
        verbose=False
    )
    print("✅ Phi-3 is awake and ready!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure 'Phi-3-mini-4k-instruct-q4.gguf' is in the 'models' folder.")

# --- 2. DATA MODELS ---
class CheckRequest(BaseModel):
    claim_text: str

# --- 3. SEARCH LOGIC ---
def web_search(query: str):
    """Searches DuckDuckGo for live evidence."""
    print(f"🌐 Searching the web for: {query}")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No recent news or evidence found."
            
            # Format results into a single string for the AI
            evidence = "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in results])
            return evidence
    except Exception as e:
        print(f"⚠️ Search failed: {e}")
        return "Search unavailable due to network timeout."

# --- 4. API ENDPOINT ---
@app.post("/verify")
def verify_claim(request: CheckRequest):
    claim = request.claim_text
    print(f"\n🔍 Fact-checking: {claim}")
    
    # Step A: Get live data
    evidence = web_search(claim)
    
    # Step B: Construct the Phi-3 Prompt
    prompt = f"""<|user|>
You are an expert Fact-Checking AI. Verify this claim using ONLY the provided evidence.

CLAIM: "{claim}"

EVIDENCE FROM WEB:
{evidence}

TASK:
1. Verdict: Is it True, False, Misleading, or Unverified?
2. Explanation: Explain why in 3,4 sentences, citing the evidence and listing the sources along with their dates. If evidence is insufficient, say "Unverified".

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
Verdict: [Your Verdict]
Explanation: [Your Explanation]<|end|>
<|assistant|>"""

    # Step C: Generate the Verdict
    start_time = time.time()
    output = llm(
        prompt, 
        max_tokens=150, 
        temperature=0.1,  # Keep it factual/deterministic
        stop=["<|end|>"], 
        echo=False
    )
    
    result_text = output['choices'][0]['text'].strip()
    print(f"⚡ Done in {round(time.time() - start_time, 2)} seconds.")
    
    return {
        "result": result_text, 
        "evidence_used": evidence
    }

# --- 5. LAUNCH ---
if __name__ == "__main__":
    import uvicorn
    # Listening on 0.0.0.0 allows your phone to connect via Wi-Fi
    print("🚀 Server starting on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)