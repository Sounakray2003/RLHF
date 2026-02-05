# ============================================================
# IMAGE CONDITION DATASET GENERATOR (BLIP + GROQ)
# ============================================================

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from dotenv import load_dotenv
from groq import Groq

# ============================================================
# FORCE .env (DO NOT USE TERMINAL VARIABLES)
# ============================================================

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ============================================================
# CONFIG
# ============================================================

IMAGE_FOLDER = r"D:/Downloads/cc_data/cc_data/images"
OUTPUT_FILE = "condition_dataset.json"

MAX_IMAGES = 50          # None = process all
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE.upper()}")

# ============================================================
# BLIP MODEL
# ============================================================

print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = "You are an expert product description writer for Amazon."

USER_PROMPT = """Create a short product description based on the image.
Only return the description. It must be concise, SEO-friendly, and suitable for mobile search.

Image: <image>"""

# ============================================================
# GROQ LLM CLIENT
# ============================================================

class SimpleLLMClient:
    def __init__(
        self,
        primary_model="llama-3.3-70b-versatile",
        fallback_model="llama3-8b-8192"
    ):
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "❌ GROQ_API_KEY missing in .env file"
            )

        print("Using GROQ_API_KEY:", api_key[:8] + "********")

        self.client = Groq(api_key=api_key)
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            return self._call(self.primary_model, messages)
        except Exception as e:
            print(f"⚠️ Primary model failed: {e}")
            print("🔁 Falling back...")
            return self._call(self.fallback_model, messages)

    def _call(self, model: str, messages) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()

# ============================================================
# HELPERS
# ============================================================

def collect_images(n: int = MAX_IMAGES) -> List[Path]:
    folder = Path(IMAGE_FOLDER)
    files = []
    for ext in VALID_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    files = sorted(set(files), key=lambda p: p.name)
    return files if n is None else files[:n]


def generate_blip_caption(img_path: Path) -> str:
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        return processor.decode(
            output[0], skip_special_tokens=True
        ).strip()
    except Exception as e:
        print(f"⚠️ BLIP failed: {img_path.name} | {e}")
        return "product item"


def rewrite_description_with_rules(
    caption: str,
    is_accepted: bool,
    llm: SimpleLLMClient
) -> str:

    if is_accepted:
        rules = """
Rewrite the product description following these rules:
- Product appears new, clean, or well maintained
- Neutral, professional Amazon listing tone
- Focus on visible condition only
- No marketing hype
- No emojis, no hashtags
- Maximum 2 short sentences
"""
    else:
        rules = """
Rewrite the product description following these rules:
- The item must NOT be described as new
- Describe visible wear, marks, scratches, or discoloration if present
- If defects are unclear, state general prior use
- Use neutral, cautious language
- Do NOT speculate about functionality
- No emojis, no hashtags
- Maximum 2 short sentences
"""

    prompt = f"""
Original image caption:
"{caption}"

{rules}

IMPORTANT:
- You MUST return a description
- Do NOT return empty text
- Do NOT invent damage
- If no defects are obvious, describe the item as used with minor wear

Return ONLY the rewritten description.
"""

    text = llm.generate(prompt).strip()

    # Safety fallback
    if not text:
        return (
            "Item shows signs of prior use and does not appear to be in new condition."
        )

    return text


def create_conversation(
    rel_image: str,
    description: str,
    is_accepted: bool
) -> Dict:

    decision_label = "Accepted" if is_accepted else "Rejected"

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image", "image": rel_image}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": description,
                        "decisions": decision_label
                    }
                ]
            }
        ]
    }

# ============================================================
# MAIN
# ============================================================

def main():
    images = collect_images(MAX_IMAGES)
    if not images:
        print("No images found.")
        return

    llm = SimpleLLMClient()
    dataset = []

    print(f"Found {len(images)} images.\n")

    for img_path in tqdm(images, desc="Generating dataset"):
        caption = generate_blip_caption(img_path)
        rel_path = f"images/{img_path.name}"

        # Accepted
        accepted_desc = rewrite_description_with_rules(
            caption, True, llm
        )
        dataset.append(
            create_conversation(rel_path, accepted_desc, True)
        )

        # Rejected
        rejected_desc = rewrite_description_with_rules(
            caption, False, llm
        )
        dataset.append(
            create_conversation(rel_path, rejected_desc, False)
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
