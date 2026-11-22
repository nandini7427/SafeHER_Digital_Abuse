from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import tempfile
import numpy as np
import hashlib

# ---- APP ----
app = FastAPI()

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- HOME ROUTE ----
@app.get("/")
def home():
    return {"message": "GuardianAI running"}

# ---- MODELS ----
text_model = pipeline("text-classification", model="unitary/toxic-bert")
image_model = pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector")

# ---- TEXT ANALYSIS ----
@app.post("/check-text")
def check_text(text: str):

    result = text_model(text)[0]

    label = result['label'].lower()
    score = float(result['score'])

    text_lower = text.lower()

    # ---- TOXICITY ----
    toxicity = score if "toxic" in label else score * 0.3

    # ---- INSULT SCORE (dynamic) ----
    insult_words = ["idiot", "stupid", "useless", "dumb", "loser"]
    insult = 0

    for word in insult_words:
        if word in text_lower:
            insult += 0.3

    insult = min(insult + (toxicity * 0.5), 1.0)

    # ---- THREAT SCORE (dynamic) ----
    threat_words = ["kill", "destroy", "attack", "die", "bomb", "shoot"]
    threat = 0

    for word in threat_words:
        if word in text_lower:
            threat += 0.4

    # sentence-based boost
    if "i will" in text_lower or "i'm going to" in text_lower:
        threat += 0.3

    threat = min(threat + (toxicity * 0.4), 1.0)

    # ---- FINAL DECISION ----
    if threat > 0.6:
        decision = "BLOCKED"
    elif insult > 0.5 or toxicity > 0.7:
        decision = "BLOCKED"
    elif toxicity > 0.4:
        decision = "WARNING"
    else:
        decision = "SAFE"

    return {
        "toxicity": round(toxicity, 3),
        "insult": round(insult, 3),
        "threat": round(threat, 3),
        "decision": decision
    }
# ---- IMAGE ANALYSIS ----
@app.post("/check-image")
async def check_image(file: UploadFile = File(...)):

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    content = await file.read()
    temp.write(content)
    temp.close()

    image = Image.open(temp.name).convert("RGB")

    # NSFW detection
    results = image_model(image)
    scores = {r['label'].lower(): r['score'] for r in results}
    nsfw_score = scores.get("nsfw", 0)

    # Deepfake hint (simple logic)
    img_np = np.array(image)
    variance = np.var(img_np)

    if variance < 300:
        deepfake_score = 0.7
    else:
        deepfake_score = 0.2

    # Decision
    if nsfw_score > 0.6:
        decision = "BLOCKED"
    elif deepfake_score > 0.6:
        decision = "SUSPICIOUS"
    else:
        decision = "SAFE"

    # Evidence hash
    evidence_hash = hashlib.md5(content).hexdigest()[:10]

    return {
        "nudity_score": round(nsfw_score, 3),
        "deepfake_score": deepfake_score,
        "decision": decision,
        "evidence_hash": evidence_hash
    }