from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from deep_translator import GoogleTranslator
import requests, json, os
from dotenv import load_dotenv
import os
from groq import Groq  # pip install groq

# ---------------- FastAPI setup ----------------
app = FastAPI(title="Humanizer WebApp")
templates = Jinja2Templates(directory="templates")

# ---------------- Groq setup ----------------
load_dotenv()  # local .env ko load karega

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_HUMANIZER = """
You are a human writer. Rewrite the text in natural, simple English 
so it looks like a real person wrote it, not AI. 

Rules:
- Use everyday words, not advanced or academic terms.
- Keep sentences short and mixed (some short, some long).
- Avoid perfect grammar — allow light imperfections.
- Keep the meaning the same but make it sound casual and human.
"""



SYSTEM_GRAMMAR = "Fix grammar only. Keep the exact meaning. Do not change word choice or sentence structure."

# ---------------- Translation ----------------
def translate(text, src, target):
    try:
        return GoogleTranslator(source=src, target=target).translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text

# ---------------- AI Detector ----------------
def detect_ai(text: str) -> bool:
    """Returns True if the text passes as human"""
    try:
        url = "https://api.zerogpt.com/api/detect/detectText"
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"input_text": text})
        resp = requests.post(url, data=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return bool(int(data["data"]["isHuman"]))  # True if human
        return False
    except Exception as e:
        print("Detection error:", e)
        return False

# ---------------- Groq Functions ----------------
def groq_rewrite(text: str, system_msg: str, temperature=1.3, model="llama-3.1-8b-instant"):
    """Humanize text using Groq"""
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": text}],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("Groq error:", e)
        return text

def grammar_fix(text: str) -> str:
    return groq_rewrite(text, SYSTEM_GRAMMAR, temperature=0.8)

# ---------------- Humanize Pipeline ----------------
def humanize_pipeline(text: str, src_lang="en", target_lang="ar", max_retries=3):
    """Translate → humanize → detect AI → retry → grammar fix"""
    final_text = text
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        print(f"[Groq Retry {attempt}]")

        # 1. Forward/back translation to naturalize (optional)
        translated = translate(final_text, src_lang, target_lang)
        final_text = translate(translated, target_lang, src_lang)

        # 2. Humanize using Groq with higher temperature
        final_text = groq_rewrite(
            final_text,
            SYSTEM_HUMANIZER + "\n\nExtra rules:\n- Break long sentences naturally\n- Use simple language\n- Maintain exact meaning",
            temperature=1.6
        )

        # 3. Detect AI
        if detect_ai(final_text):
            print("✅ Passed AI detector")
            break
        else:
            print("❌ Still AI-like, retrying...")

    # 4. Grammar fix for final polish
    final_text = grammar_fix(final_text)
    return final_text

# ---------------- Web Endpoints ----------------
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/process", response_class=HTMLResponse)
async def process_text(request: Request,
                       text: str = Form(...),
                       src_lang: str = Form("en"),
                       target_lang: str = Form("fr")):
    original = text
    final_text = humanize_pipeline(original, src_lang, target_lang)
    is_human = detect_ai(final_text)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {
            "original": original,
            "final_text": final_text,
            "is_human": is_human
        }
    })
