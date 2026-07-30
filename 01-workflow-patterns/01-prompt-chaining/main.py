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

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

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

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
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
    system = (
        f"You are a professional translator specialising in {target_language}. "
        "Translate the user's text exactly, preserving meaning and tone. "
        "Output ONLY the translated text — no explanations, no quotes, no alternatives."
    )
    prompt = (
        f"Translate the following English text into {target_language}.\n"
        f"Example: 'The weather is nice today.' → 'Il fait beau aujourd'hui.' (for French)\n\n"
        f"Now translate this:\n{text}"
    )
    return llm_call(prompt, system)


def gate_verify_translation(original: str, translation: str, target_language: str) -> bool:
    """Gate: Verify that the translation is accurate. Returns True if acceptable."""
    system = (
        "You are a strict translation quality reviewer. "
        "Your only job is to respond with the single word PASS or FAIL. "
        "Do not write anything else."
    )
    prompt = (
        f"Does this {target_language} translation accurately convey the meaning of the original?"
        f"\n\nOriginal (English):\n{original}"
        f"\n\n{target_language} translation:\n{translation}"
        "\n\nRespond with PASS if the translation is accurate, or FAIL if it is wrong or unnatural."
        "\nYour response must be exactly one word: PASS or FAIL."
    )
    result = llm_call(prompt, system).strip().upper()
    return result.startswith("PASS")


def step_polish_translation(translation: str, target_language: str) -> str:
    """Step 2: Polish the translation for natural flow."""
    system = (
        f"You are a native {target_language} editor. "
        "Refine the given translation so it sounds fluent and natural. "
        "Output only the polished text with no explanation."
    )
    prompt = f"Polish this {target_language} translation:\n\n{translation}"
    return llm_call(prompt, system)


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
    # Note: for best multilingual results use a larger model, e.g. ollama pull mistral
    english_text = "Artificial intelligence is transforming the way software is built and maintained."
    run_translation_chain(english_text, "French")
