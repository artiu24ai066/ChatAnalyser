# =============================================================================
# nlp_analyzer.py
# =============================================================================
# This module handles ALL NLP-related processing for the WhatsApp Chat Analyzer.
#
# Responsibilities:
#   - Load pretrained transformer models (once, with caching)
#   - Decide which messages are valid for analysis
#   - Extract emojis from messages
#   - Run batch sentiment inference
#   - Run batch emotion inference
#   - Return an enriched DataFrame with new NLP columns
#
# This module does NOT touch the UI (that lives in app.py).
# This module does NOT change existing DataFrame columns.
# It only ADDS new columns.
#
# Models used:
#   Sentiment : shae2977/xlm-roberta-hinglish-sentiment-analysis
#               XLM-RoBERTa fine-tuned on Hinglish/code-mixed text.
#               Labels: Positive, Neutral, Negative
#
#   Emotion   : tabularisai/multilingual-emotion-classification
#               XLM-R based multilingual model.
#               Labels: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
#               Added in Phase 2 — placeholder loader is here already.
#
# Performance notes:
#   - Models are loaded ONCE using Streamlit's @st.cache_resource.
#     If you call load_sentiment_model() ten times, the model is only
#     downloaded/loaded the very first time. After that it is reused.
#   - Inference is done in BATCHES, not message-by-message.
#     This is much faster for large chat exports.
#   - Invalid messages (media, group notifications, empty) are SKIPPED
#     and get NaN in the sentiment/emotion columns.
# =============================================================================

import re
import streamlit as st
import pandas as pd
import numpy as np
import emoji

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Hugging Face model identifiers
SENTIMENT_MODEL_NAME = "shae2977/xlm-roberta-hinglish-sentiment-analysis"
EMOTION_MODEL_NAME   = "tabularisai/multilingual-emotion-classification"

# ── Label mapping for the sentiment model ────────────────────────────────────
# This model's config.json does not set id2label correctly, so it returns
# raw numeric labels: LABEL_0, LABEL_1, LABEL_2.
# We verified the correct mapping from the model card and test output:
#   LABEL_0 → Negative  (e.g. "bhai kya bakwas hai" → LABEL_0, which is negative)
#   LABEL_1 → Neutral   (e.g. "okay fine", "Meeting at 5pm" → LABEL_1)
#   LABEL_2 → Positive  (e.g. "bahut maza aaya aaj!" → LABEL_2, which is positive)
# If the model already returns human-readable labels (on some environments it does),
# the mapping below is a no-op for those labels and only fixes the raw ones.
SENTIMENT_LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    # pass-through for environments where the model returns readable labels
    "Negative": "Negative",
    "Neutral":  "Neutral",
    "Positive": "Positive",
}

# ── Label mapping for the emotion model ──────────────────────────────────────
# tabularisai/multilingual-emotion-classification returns readable labels
# directly from its id2label config. We still define a passthrough map so
# that:
#   (a) the code is consistent with the sentiment model pattern
#   (b) any unexpected raw LABEL_N values get passed through unchanged
#       rather than crashing the app
#
# Confirmed label set (7 classes):
#   Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
#
# Each label also gets an emoji icon used in charts for quick visual scanning.
EMOTION_LABEL_MAP = {
    # Model returns all-lowercase labels — map every one to Title Case
    # so they match EMOTION_COLORS and EMOTION_ICONS keys exactly.
    # Full label set confirmed from model's id2label config (11 classes):
    "anger":       "Anger",
    "contempt":    "Contempt",
    "disgust":     "Disgust",
    "fear":        "Fear",
    "frustration": "Frustration",
    "gratitude":   "Gratitude",
    "joy":         "Joy",
    "love":        "Love",
    "neutral":     "Neutral",
    "sadness":     "Sadness",
    "surprise":    "Surprise",
    # Pass-through for Title Case (future-proofing)
    "Anger":       "Anger",
    "Contempt":    "Contempt",
    "Disgust":     "Disgust",
    "Fear":        "Fear",
    "Frustration": "Frustration",
    "Gratitude":   "Gratitude",
    "Joy":         "Joy",
    "Love":        "Love",
    "Neutral":     "Neutral",
    "Sadness":     "Sadness",
    "Surprise":    "Surprise",
}

# Colour palette — one distinct colour per emotion
EMOTION_COLORS = {
    "Joy":         "#FFD700",   # gold
    "Love":        "#FF69B4",   # hot pink
    "Gratitude":   "#32CD32",   # lime green
    "Surprise":    "#FF8C00",   # dark orange
    "Neutral":     "#708090",   # slate grey
    "Sadness":     "#1E90FF",   # dodger blue
    "Frustration": "#FF6347",   # tomato orange-red
    "Anger":       "#FF2400",   # scarlet red
    "Fear":        "#9400D3",   # dark violet
    "Contempt":    "#8B4513",   # saddle brown
    "Disgust":     "#228B22",   # forest green
}

# Emoji icon per emotion — purely cosmetic for UI display
EMOTION_ICONS = {
    "Joy":         "😄",
    "Love":        "❤️",
    "Gratitude":   "🙏",
    "Surprise":    "😮",
    "Neutral":     "😐",
    "Sadness":     "😢",
    "Frustration": "😤",
    "Anger":       "😡",
    "Fear":        "😨",
    "Contempt":    "😒",
    "Disgust":     "🤢",
}

# How many messages to send to the model at once.
# Larger batches = faster overall, but use more RAM.
# 32 is a safe default for most laptops.
BATCH_SIZE = 128  # send 128 messages per model call — faster than 64 on CPU

MAX_LENGTH = 64   # reduced from 128 — WhatsApp messages rarely exceed 30 tokens,
                  # cutting this in half roughly halves the computation per batch

# Messages that should NOT be analyzed (they carry no real sentiment).
# We match these as substrings (case-insensitive).
SKIP_PATTERNS = [
    r"<media omitted>",          # images, videos, audio
    r"<image omitted>",
    r"<video omitted>",
    r"<audio omitted>",
    r"<sticker omitted>",
    r"<document omitted>",
    r"<gif omitted>",
    r"this message was deleted",
    r"you deleted this message",
    r"messages and calls are end-to-end encrypted",
    r"missed voice call",
    r"missed video call",
]

# Compiled regex for speed (compiled once at import time)
_SKIP_RE = re.compile("|".join(SKIP_PATTERNS), flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# MODEL LOADERS  (cached so they run only once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading sentiment model (first time only)...")
def load_sentiment_model():
    """
    Load the Hinglish XLM-RoBERTa sentiment pipeline from Hugging Face.

    This function is decorated with @st.cache_resource which means:
      - First call  : downloads the model (~1 GB), loads it into memory
      - Every call after : instantly returns the already-loaded pipeline

    The pipeline object handles tokenization + inference internally.
    We just pass it a list of strings and get back labels + scores.

    Returns
    -------
    transformers.Pipeline or None
        The loaded pipeline, or None if loading fails (with a Streamlit error).
    """
    try:
        # Import here so that if transformers is not installed,
        # the rest of the app still works (just NLP features won't).
        from transformers import pipeline

        sentiment_pipe = pipeline(
            task="text-classification",
            model=SENTIMENT_MODEL_NAME,
            tokenizer=SENTIMENT_MODEL_NAME,
            max_length=MAX_LENGTH,
            truncation=True,       # silently truncate messages > MAX_LENGTH tokens
            padding=True,
        )
        return sentiment_pipe

    except Exception as e:
        st.error(
            f"Could not load sentiment model '{SENTIMENT_MODEL_NAME}'.\n"
            f"Error: {e}\n\n"
            "Make sure you have installed: transformers, torch, sentencepiece\n"
            "Run:  pip install transformers torch sentencepiece"
        )
        return None


@st.cache_resource(show_spinner="Loading emotion model (first time only)...")
def load_emotion_model():
    """
    Load the multilingual emotion classification pipeline.
    This will be used in Phase 2.

    Decorated with @st.cache_resource for the same reason as above.

    Returns
    -------
    transformers.Pipeline or None
    """
    try:
        from transformers import pipeline

        emotion_pipe = pipeline(
            task="text-classification",
            model=EMOTION_MODEL_NAME,
            tokenizer=EMOTION_MODEL_NAME,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        )
        return emotion_pipe

    except Exception as e:
        st.error(
            f"Could not load emotion model '{EMOTION_MODEL_NAME}'.\n"
            f"Error: {e}"
        )
        return None


# ---------------------------------------------------------------------------
# MESSAGE FILTERING
# ---------------------------------------------------------------------------

def is_analyzable(message: str) -> bool:
    """
    Decide whether a message should be sent to the sentiment/emotion model.

    We skip:
      - None / NaN values
      - Empty strings or whitespace-only strings
      - Group notifications (e.g. "Alice added Bob")
      - Media-only messages ("<Media omitted>")
      - System messages ("Messages and calls are end-to-end encrypted")

    Parameters
    ----------
    message : str
        The raw message text from the DataFrame.

    Returns
    -------
    bool
        True  → send this message to the model
        False → skip it (set NaN in output columns)
    """
    # Handle None / float NaN
    if not isinstance(message, str):
        return False

    stripped = message.strip()

    # Empty or whitespace only
    if len(stripped) == 0:
        return False

    # Matches any skip pattern
    if _SKIP_RE.search(stripped):
        return False

    return True


# ---------------------------------------------------------------------------
# EMOJI EXTRACTION
# ---------------------------------------------------------------------------

def extract_emojis_from_text(text: str) -> list:
    """
    Extract all emojis from a single message string.

    We support both old and new versions of the `emoji` library:
      - emoji >= 2.0 uses emoji.emoji_list(text) which returns dicts
      - emoji < 2.0  uses emoji.EMOJI_DATA as a lookup set

    Parameters
    ----------
    text : str

    Returns
    -------
    list of str
        Example: ["😂", "❤️"]
        Empty list if no emojis found or text is not a string.
    """
    if not isinstance(text, str):
        return []

    # emoji >= 2.0 API
    if hasattr(emoji, "emoji_list"):
        return [item["emoji"] for item in emoji.emoji_list(text)]

    # emoji < 2.0 API (fallback)
    if hasattr(emoji, "EMOJI_DATA"):
        return [char for char in text if char in emoji.EMOJI_DATA]

    return []


def add_emoji_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two new columns to the DataFrame:
      - extracted_emojis : list of emoji characters found in the message
      - emoji_count      : integer count of emojis in the message

    This is safe to call on the whole DataFrame including media/notification
    rows because we are just extracting characters, not running a model.

    Parameters
    ----------
    df : pd.DataFrame  (must have a 'message' column)

    Returns
    -------
    pd.DataFrame  with two new columns added in-place (returns df for chaining)
    """
    df = df.copy()  # never mutate the caller's DataFrame
    df["extracted_emojis"] = df["message"].apply(extract_emojis_from_text)
    df["emoji_count"]       = df["extracted_emojis"].apply(len)
    return df


# ---------------------------------------------------------------------------
# BATCH INFERENCE HELPERS
# ---------------------------------------------------------------------------

def _run_pipeline_in_batches(pipe, texts: list, batch_size: int = BATCH_SIZE,
                             progress_bar=None, progress_offset: float = 0.0,
                             progress_range: float = 1.0) -> list:
    """
    Run a HuggingFace text-classification pipeline over a list of texts
    in batches instead of one message at a time.

    WHY BATCHES?
    ------------
    Calling pipe(text) in a Python for-loop sends one sentence at a time
    to the model. This is slow because the GPU/CPU is mostly idle between
    calls. Sending BATCH_SIZE sentences at once lets the hardware process
    them in parallel, which is significantly faster for large chat exports.

    Parameters
    ----------
    pipe             : transformers.Pipeline
    texts            : list of str  — only the messages we want to analyze
    batch_size       : int          — how many at once (default BATCH_SIZE)
    progress_bar     : st.progress object or None — Streamlit progress bar to update
    progress_offset  : float  — starting fraction for this model (0.0 for sentiment, 0.5 for emotion)
    progress_range   : float  — fraction of the bar this model covers (0.5 each if two models)

    Returns
    -------
    list of dict
        Each dict has keys: 'label' (str) and 'score' (float).
        Example: [{'label': 'Positive', 'score': 0.94}, ...]
    """
    results = []
    total   = len(texts)

    for start in range(0, total, batch_size):
        batch        = texts[start : start + batch_size]
        batch_results= pipe(batch)
        results.extend(batch_results)

        # Update Streamlit progress bar if one was passed in
        if progress_bar is not None and total > 0:
            done_fraction    = min(start + batch_size, total) / total
            overall_fraction = progress_offset + done_fraction * progress_range
            progress_bar.progress(
                min(overall_fraction, 1.0),
                text=f"Analyzed {min(start + batch_size, total):,} / {total:,} messages"
            )

    return results


# ---------------------------------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------

def run_sentiment_analysis(df: pd.DataFrame, pipe, progress_bar=None,
                           progress_offset: float = 0.0,
                           progress_range: float = 0.5) -> pd.DataFrame:
    """
    Add 'sentiment' and 'sentiment_score' columns to the DataFrame.
    """
    df = df.copy()
    df["sentiment"]       = None
    df["sentiment_score"] = None

    mask = df["message"].apply(is_analyzable)
    valid_indices  = df.index[mask].tolist()
    valid_messages = df.loc[valid_indices, "message"].tolist()

    if len(valid_messages) == 0:
        return df

    results = _run_pipeline_in_batches(
        pipe, valid_messages, batch_size=BATCH_SIZE,
        progress_bar=progress_bar,
        progress_offset=progress_offset,
        progress_range=progress_range
    )

    for idx, result in zip(valid_indices, results):
        raw_label    = result["label"]
        mapped_label = SENTIMENT_LABEL_MAP.get(raw_label, raw_label)
        df.at[idx, "sentiment"]       = mapped_label
        df.at[idx, "sentiment_score"] = round(result["score"], 4)

    return df


# ---------------------------------------------------------------------------
# EMOTION ANALYSIS  (Phase 2 — function is ready, call it in Phase 2)
# ---------------------------------------------------------------------------

def run_emotion_analysis(df: pd.DataFrame, pipe, progress_bar=None,
                         progress_offset: float = 0.33,
                         progress_range: float = 0.33) -> pd.DataFrame:
    """
    Add 'emotion' and 'emotion_score' columns to the DataFrame.
    """
    df = df.copy()
    df["emotion"]       = None
    df["emotion_score"] = None

    mask = df["message"].apply(is_analyzable)
    valid_indices  = df.index[mask].tolist()
    valid_messages = df.loc[valid_indices, "message"].tolist()

    if len(valid_messages) == 0:
        return df

    results = _run_pipeline_in_batches(
        pipe, valid_messages, batch_size=BATCH_SIZE,
        progress_bar=progress_bar,
        progress_offset=progress_offset,
        progress_range=progress_range
    )

    for idx, result in zip(valid_indices, results):
        raw_label    = result["label"]
        mapped_label = EMOTION_LABEL_MAP.get(raw_label, raw_label)
        df.at[idx, "emotion"]       = mapped_label
        df.at[idx, "emotion_score"] = round(result["score"], 4)

    return df


# ---------------------------------------------------------------------------
# MAIN ENRICHMENT FUNCTION
# ---------------------------------------------------------------------------

def enrich_dataframe(df: pd.DataFrame, run_sentiment: bool = True,
                     run_emotion: bool = True, run_sarcasm: bool = True,
                     progress_bar=None) -> pd.DataFrame:
    """
    Orchestrator function. Takes the base DataFrame from preprocessor.py
    and returns an enriched DataFrame with all NLP columns.

    Phase 1 : adds sentiment, sentiment_score, extracted_emojis, emoji_count
    Phase 2 : adds emotion, emotion_score
    Phase 7 : adds is_sarcastic, sarcasm_score

    Progress bar is split evenly across however many models run.

    Parameters
    ----------
    df            : pd.DataFrame  (output of preprocessor.preprocess())
    run_sentiment : bool
    run_emotion   : bool
    run_sarcasm   : bool  — run sarcasm/irony detection (default True)
    progress_bar  : st.progress object or None

    Returns
    -------
    pd.DataFrame  — same as input but with new NLP columns added
    """
    import sarcasm_analyzer  # local import keeps module loading lazy

    # Count how many models will run to divide the progress bar evenly
    models_to_run = sum([run_sentiment, run_emotion, run_sarcasm])
    slice_size    = 1.0 / models_to_run if models_to_run > 0 else 1.0
    current_offset = 0.0

    # Step 1: Emoji extraction — fast, no model needed, no progress slice needed
    df = add_emoji_columns(df)

    # Step 2: Sentiment
    if run_sentiment:
        pipe = load_sentiment_model()
        if pipe is not None:
            df = run_sentiment_analysis(
                df, pipe,
                progress_bar=progress_bar,
                progress_offset=current_offset,
                progress_range=slice_size
            )
        else:
            df["sentiment"]       = None
            df["sentiment_score"] = None
        current_offset += slice_size

    # Step 3: Emotion
    if run_emotion:
        pipe = load_emotion_model()
        if pipe is not None:
            df = run_emotion_analysis(
                df, pipe,
                progress_bar=progress_bar,
                progress_offset=current_offset,
                progress_range=slice_size
            )
        else:
            df["emotion"]       = None
            df["emotion_score"] = None
        current_offset += slice_size

    # Step 4: Sarcasm (Phase 7)
    if run_sarcasm:
        df = sarcasm_analyzer.enrich_sarcasm(
            df,
            progress_bar=progress_bar,
            progress_offset=current_offset,
            progress_range=slice_size
        )

    return df


# ---------------------------------------------------------------------------
# CACHED ENRICHMENT  (use this from app.py instead of enrich_dataframe directly)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_enrich(file_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached wrapper around enrich_dataframe().

    WHY THIS EXISTS
    ---------------
    Without caching, every time the user clicks "Show Analysis" or changes
    the selected_user dropdown, Streamlit re-runs the whole script and both
    transformer models run again on every message — even if the same file
    is already loaded.

    With @st.cache_data, the result is stored keyed on file_hash.
    If the same file is uploaded again in the same session, the cached
    result is returned instantly — no inference needed.

    HOW TO USE FROM app.py
    ----------------------
    import hashlib
    file_hash = hashlib.md5(bytes_data).hexdigest()
    df = nlp_analyzer.cached_enrich(file_hash, base_df)

    Parameters
    ----------
    file_hash : str  — MD5 hex digest of the raw uploaded file bytes
                       Used as the cache key. Different files → different results.
    df        : pd.DataFrame  — output of preprocessor.preprocess()

    Returns
    -------
    pd.DataFrame  — enriched with sentiment, emotion, emoji columns
    """
    # Progress bar lives inside the cached call — shows during the one real run,
    # disappears immediately when the cached result is returned on re-use.
    progress_bar = st.progress(0.0, text="Starting NLP analysis...")
    result = enrich_dataframe(df, run_sentiment=True, run_emotion=True,
                              run_sarcasm=True, progress_bar=progress_bar)
    progress_bar.progress(1.0, text="NLP analysis complete.")
    progress_bar.empty()   # remove the bar from the UI after completion
    return result


# ---------------------------------------------------------------------------
# QUICK SELF-TEST  (run this file directly to test: python nlp_analyzer.py)
# ---------------------------------------------------------------------------

def _self_test():
    """
    Tests the sentiment model on 15 sample Hinglish/English WhatsApp messages.
    Run this directly from the terminal:

        python nlp_analyzer.py

    Expected output: a table with message, predicted sentiment, and confidence.
    This does NOT require Streamlit — it uses the model directly.
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers torch sentencepiece")
        return

    print(f"\nLoading model: {SENTIMENT_MODEL_NAME}")
    print("This may take a minute on first run (model download ~1 GB)...\n")

    try:
        pipe = pipeline(
            task="text-classification",
            model=SENTIMENT_MODEL_NAME,
            tokenizer=SENTIMENT_MODEL_NAME,
            max_length=MAX_LENGTH,
            truncation=True,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 15 test messages covering:
    #   - Pure Hindi transliterated (Hinglish)
    #   - Mixed Hindi + English (code-switching)
    #   - Pure English
    #   - Emojis
    #   - Short ambiguous messages
    #   - Negative sentiment
    #   - Positive sentiment
    test_messages = [
        "yaar aaj bahut acha laga",           # Pure Hinglish, positive
        "bhai kya bakwas hai",                 # Hinglish, negative
        "kal milte hain",                      # Neutral/informational
        "thank you yaar ❤️",                  # Positive with emoji
        "acha nahi laga yaar",                 # Negative
        "OMG ye kya tha 😭",                   # Surprise/negative with emoji
        "bahut maza aaya aaj!",                # Positive
        "yaar bahut bura hua",                 # Negative
        "okay fine",                           # Ambiguous/neutral short
        "haha 😂😂😂",                         # Positive (laughter)
        "I am so happy today!",                # Pure English, positive
        "This is terrible.",                   # Pure English, negative
        "Meeting at 5pm.",                     # Neutral/informational
        "bhaiiii 😂😂",                        # Positive Hinglish
        "sab theek hai",                       # Neutral Hinglish
    ]

    print(f"{'Message':<40} {'Sentiment':<12} {'Confidence'}")
    print("-" * 68)

    results = pipe(test_messages, batch_size=8)

    for msg, res in zip(test_messages, results):
        raw_label   = res["label"]
        mapped      = SENTIMENT_LABEL_MAP.get(raw_label, raw_label)
        score       = res["score"]
        print(f"{msg:<40} {mapped:<12} {score:.4f}  (raw: {raw_label})")

    print("\nSelf-test complete.")
    print("\nWhat to check:")
    print("  - Positive messages should get 'Positive' with score > 0.7")
    print("  - Negative messages should get 'Negative' with score > 0.7")
    print("  - 'kal milte hain', 'okay fine', 'Meeting at 5pm' → likely Neutral")
    print("  - Short/ambiguous messages may have lower confidence scores")
    print("  - This is expected — Hinglish is informal and context-dependent")


if __name__ == "__main__":
    _self_test()
