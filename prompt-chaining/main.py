"""
Prompt Chaining Pattern with OpenAI

Prompt chaining decomposes a task into a sequence of steps where each LLM call
processes the output of the previous one. This is useful for tasks that benefit
from a structured pipeline, like:
  - Research → Draft → Edit
  - Translate → Verify → Refine
  - Extract → Validate → Format

Each step in the chain can apply its own focused prompt, and you can add
gate/validation logic between steps.
"""

import os
import sys
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError

load_dotenv(find_dotenv())

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def llm_call(prompt: str, system: str = "", model: str = "llama3.2") -> str:
    """Single LLM call returning the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except RateLimitError as e:
        if "insufficient_quota" in str(e):
            sys.exit(
                "Error: OpenAI quota exceeded. Add billing credits at "
                "https://platform.openai.com/settings/billing"
            )
        raise
    except AuthenticationError:
        sys.exit("Error: Invalid OPENAI_API_KEY. Check your .env file.")
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Example chain: Article generation pipeline
# Step 1 → Generate an outline
# Step 2 → Write a draft from the outline
# Step 3 → Edit the draft for clarity and conciseness
# ---------------------------------------------------------------------------

def step_generate_outline(topic: str) -> str:
    """Step 1: Generate a structured outline for the given topic."""
    system = "You are an expert content strategist. Create clear, structured outlines."
    prompt = f"Create a concise outline (3-5 main points) for an article about: {topic}"
    outline = llm_call(prompt, system)
    print("=== Step 1: Outline ===")
    print(outline)
    print()
    return outline


def step_write_draft(outline: str) -> str:
    """Step 2: Write a draft article based on the outline."""
    system = "You are a skilled writer. Write engaging, informative content."
    prompt = f"Write a short article (2-3 paragraphs) based on this outline:\n\n{outline}"
    draft = llm_call(prompt, system)
    print("=== Step 2: Draft ===")
    print(draft)
    print()
    return draft


def step_edit_draft(draft: str) -> str:
    """Step 3: Edit the draft for clarity, tone and conciseness."""
    system = "You are a professional editor. Improve clarity, fix grammar, and tighten prose."
    prompt = f"Edit the following article for clarity and conciseness. Return only the edited version:\n\n{draft}"
    edited = llm_call(prompt, system)
    print("=== Step 3: Edited Article ===")
    print(edited)
    print()
    return edited


def run_article_chain(topic: str) -> str:
    """Run the full prompt chain: outline → draft → edit."""
    outline = step_generate_outline(topic)
    draft = step_write_draft(outline)
    final = step_edit_draft(draft)
    return final


# ---------------------------------------------------------------------------
# Example chain with gate: Translation pipeline with quality check
# Step 1 → Translate text
# Gate  → Verify translation quality (if score < threshold, retry)
# Step 2 → Polish the translation
# ---------------------------------------------------------------------------

def step_translate(text: str, target_language: str) -> str:
    """Step 1: Translate text into the target language."""
    prompt = f"Translate the following text to {target_language}. Return only the translation:\n\n{text}"
    return llm_call(prompt)


def gate_verify_translation(original: str, translation: str, target_language: str) -> bool:
    """Gate: Verify that the translation is accurate. Returns True if acceptable."""
    system = "You are a translation quality reviewer. Respond with only 'PASS' or 'FAIL'."
    prompt = (
        f"Original text:\n{original}\n\n"
        f"Translation ({target_language}):\n{translation}\n\n"
        "Is this translation accurate and natural? Respond with only PASS or FAIL."
    )
    result = llm_call(prompt, system)
    return "PASS" in result.upper()


def step_polish_translation(translation: str, target_language: str) -> str:
    """Step 2: Polish the translation for natural flow."""
    prompt = (
        f"Polish this {target_language} translation to sound natural to a native speaker. "
        f"Return only the polished version:\n\n{translation}"
    )
    return llm_call(prompt)


def run_translation_chain(text: str, target_language: str, max_retries: int = 2) -> str:
    """Run the translation chain with a quality gate."""
    print(f"=== Translating to {target_language} ===")
    print(f"Original: {text}\n")

    for attempt in range(max_retries + 1):
        translation = step_translate(text, target_language)
        print(f"Attempt {attempt + 1} translation: {translation}")

        if gate_verify_translation(text, translation, target_language):
            print("Gate: PASSED\n")
            break
        print(f"Gate: FAILED — retrying (attempt {attempt + 1}/{max_retries})\n")
    else:
        print("Gate: Max retries reached, proceeding with last translation\n")

    polished = step_polish_translation(translation, target_language)
    print(f"Polished: {polished}")
    return polished


if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT CHAINING DEMO")
    print("=" * 60)
    print()

    # Demo 1: Article generation chain
    print("--- Demo 1: Article Generation Chain ---\n")
    final_article = run_article_chain("the impact of AI on software engineering")
    print("Final article produced.\n")

    print("-" * 60)

    # Demo 2: Translation chain with gate
    print("\n--- Demo 2: Translation Chain with Quality Gate ---\n")
    english_text = "The early bird catches the worm, but the second mouse gets the cheese."
    run_translation_chain(english_text, "French")
