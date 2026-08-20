# =============================================================================
# sarcasm_analyzer.py
# =============================================================================
# Phase 7 — Sarcasm / Irony Detection
#
# This module detects sarcasm and irony in WhatsApp messages.
#
# Model used:
#   amaan00z/sarcasm_xlmr
#   - XLM-RoBERTa base fine-tuned specifically for sarcasm detection
#   - XLM-RoBERTa was pretrained on 100 languages including multilingual text, which gives it better coverage of Hinglish code-switching compared to English-only models like cardiffnlp/twitter-roberta-base-irony
#   - Binary classifier: LABEL_0 (not sarcastic) | LABEL_1 (sarcastic)
#   - Label mapping confirmed empirically by probing on Hinglish test messages
#
# Confirmed Hinglish sarcasm detection (from probe test):
#   "haan bilkul, tune toh kamaal kar diya" → Sarcastic (0.999) ✓
#   "wah wah, kya kaam kiya tune"           → Sarcastic (0.909) ✓
#   "aaj bahut acha laga yaar"              → Not Sarcastic (0.858) ✓
#   "bahut maza aaya aaj"                   → Not Sarcastic (0.943) ✓
#
# Important limitations (honest):
#   1. The base XLM-RoBERTa multilingual pretraining helps with Hinglish, but the fine-tuning dataset composition for this specific model is not fully documented — treat results as a useful statistical signal.
#   2. Context-dependent sarcasm that spans multiple messages cannot be detected because each message is classified independently.
#   3. Sarcasm score is model confidence, not a human sarcasm intensity rating.
#   4. Low scores near 0.5 indicate the model is uncertain.
#   5. These predictions are statistical estimates, not ground truth.
#
# Output columns added to the DataFrame:
#   is_sarcastic   : bool/None  — True if model predicts sarcasm
#   sarcasm_confidence  : float/None — confidence score (0.0–1.0) for
#                                 the predicted label
#
# Architecture note:
#   This module is INTENTIONALLY separate from nlp_analyzer.py.
#   Sarcasm is a higher-order signal that sits on top of sentiment
#   and emotion. The pipeline is:
#
#     Message → Sentiment → Emotion → Sarcasm
#
#   By keeping sarcasm in its own module, future model swaps or
#   Hinglish-specific fine-tuning remain isolated from the rest.
# =============================================================================

import re
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SARCASM_MODEL_NAME = "amaan00z/sarcasm_xlmr"

# Label map — confirmed by probing the model with known sarcastic/genuine messages.
# The model's config.json does not name the labels, so we determined the mapping
# empirically:
#
#   LABEL_0 → Not Sarcastic  (non-sarcastic/genuine messages)
#   LABEL_1 → Sarcastic      (ironic/sarcastic messages)
#
# Hinglish test results that confirmed this:
#   "haan bilkul, tune toh kamaal kar diya" → LABEL_1 (0.999) ✓ sarcastic
#   "wah wah, kya kaam kiya tune"           → LABEL_1 (0.909) ✓ sarcastic
#   "aaj bahut acha laga yaar"              → LABEL_0 (0.858) ✓ genuine
#   "bahut maza aaya aaj"                   → LABEL_0 (0.943) ✓ genuine
#
# This XLM-RoBERTa-based model was selected because its multilingual
# pretraining provides broader coverage for multilingual and code-mixed
# text such as Hinglish compared with English-focused irony models.
SARCASM_LABEL_MAP = {
    "LABEL_0": False,   # Not Sarcastic
    "LABEL_1": True,    # Sarcastic
    # pass-through in case a future version uses readable labels
    "sarcasm":     True,
    "non_sarcasm": False,
}

# Batch size for inference — same reasoning as nlp_analyzer.py
BATCH_SIZE = 128

# Max token length — tweets are short, WhatsApp messages similarly so
MAX_LENGTH = 64

# Messages to skip — same patterns as nlp_analyzer.is_analyzable()
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
    r"security code changed",
    r"https?://\S+",             
]

_SKIP_RE = re.compile("|".join(SKIP_PATTERNS), flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# MODEL LOADER  (cached — loads once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading sarcasm detection model (first time only)...")
def load_sarcasm_model():
    """
    Load the irony/sarcasm detection pipeline from Hugging Face.

    Decorated with @st.cache_resource — the model is downloaded and loaded
    into memory exactly once per Streamlit session. Every subsequent call
    returns the already-loaded pipeline instantly.

    Returns
    -------
    transformers.Pipeline or None
        None if the model fails to load (error shown in Streamlit UI).
    """
    try:
        from transformers import pipeline

        sarcasm_pipe = pipeline(
            task="text-classification",
            model=SARCASM_MODEL_NAME,
            tokenizer=SARCASM_MODEL_NAME,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        )
        return sarcasm_pipe

    except Exception as e:
        st.error(
            f"Could not load sarcasm model '{SARCASM_MODEL_NAME}'.\n"
            f"Error: {e}\n\n"
            "Make sure transformers and torch are installed."
        )
        return None


# ---------------------------------------------------------------------------
# MESSAGE FILTERING
# ---------------------------------------------------------------------------

def is_analyzable(message: str) -> bool:
    """
    Decide whether a message should be sent to the sarcasm model.
    Identical logic to nlp_analyzer.is_analyzable() — kept here so
    sarcasm_analyzer.py is fully self-contained and does not depend
    on importing nlp_analyzer (avoids circular imports).
    """
    if not isinstance(message, str):
        return False
    stripped = message.strip()
    if len(stripped) == 0:
        return False
    if _SKIP_RE.search(stripped):
        return False
    return True


# ---------------------------------------------------------------------------
# BATCH INFERENCE
# ---------------------------------------------------------------------------

def _run_sarcasm_in_batches(pipe, texts: list,
                            progress_bar=None,
                            progress_offset: float = 0.0,
                            progress_range: float = 1.0) -> list:
    """
    Run sarcasm inference in batches of BATCH_SIZE messages.

    Parameters
    ----------
    pipe            : transformers.Pipeline
    texts           : list of str
    progress_bar    : st.progress object or None
    progress_offset : float  — where in the overall progress bar to start
    progress_range  : float  — what fraction of the bar sarcasm covers

    Returns
    -------
    list of dict — each dict has 'label' (str) and 'score' (float)
    """
    results = []
    total   = len(texts)

    for start in range(0, total, BATCH_SIZE):
        batch         = texts[start : start + BATCH_SIZE]
        batch_results = pipe(batch)
        results.extend(batch_results)

        if progress_bar is not None and total > 0:
            done_fraction    = min(start + BATCH_SIZE, total) / total
            overall_fraction = progress_offset + done_fraction * progress_range
            progress_bar.progress(
                min(overall_fraction, 1.0),
                text=f"Sarcasm: {min(start + BATCH_SIZE, total):,} / {total:,} messages"
            )

    return results


# ---------------------------------------------------------------------------
# MAIN ENRICHMENT FUNCTION
# ---------------------------------------------------------------------------

def run_sarcasm_analysis(df: pd.DataFrame, pipe,
                         progress_bar=None,
                         progress_offset: float = 0.0,
                         progress_range: float = 1.0) -> pd.DataFrame:
    """
    Add 'is_sarcastic' and 'sarcasm_confidence' columns to the DataFrame.

    Steps:
    1. Identify analyzable messages (skip media, notifications, empty)
    2. Run batch inference
    3. Map raw labels → bool (irony=True, non_irony=False)
    4. Write results back to the correct rows
    5. Skipped rows get None

    Parameters
    ----------
    df              : pd.DataFrame  (must have 'message' column)
    pipe            : loaded sarcasm pipeline
    progress_bar    : st.progress object or None
    progress_offset : float
    progress_range  : float

    Returns
    -------
    pd.DataFrame with 'is_sarcastic' (bool/None) and 'sarcasm_confidence' (float/None)
    """
    df = df.copy()

    # Initialise with None — keeps dtype as object so bools can be stored
    df["is_sarcastic"]  = None
    df["sarcasm_confidence"] = None

    mask = df["message"].apply(is_analyzable)
    valid_indices = df.index[mask].tolist()
    valid_messages = df.loc[valid_indices, "message"].tolist()

    if len(valid_messages) == 0:
        return df

    results = _run_sarcasm_in_batches(
        pipe, valid_messages,
        progress_bar=progress_bar,
        progress_offset=progress_offset,
        progress_range=progress_range
    )

    labels = []
    confidences = []

    for result in results:
        raw_label = result["label"]

        if raw_label not in SARCASM_LABEL_MAP:
            raise ValueError(
                f"Unexpected sarcasm model label: {raw_label}"
            )

        labels.append(SARCASM_LABEL_MAP[raw_label])
        confidences.append(round(result["score"], 4))

    df.loc[valid_indices, "is_sarcastic"] = labels
    df.loc[valid_indices, "sarcasm_confidence"] = confidences

    return df


# ---------------------------------------------------------------------------
# CACHED ENRICHMENT ENTRY POINT
# ---------------------------------------------------------------------------

def enrich_sarcasm(df: pd.DataFrame,
                   progress_bar=None,
                   progress_offset: float = 0.0,
                   progress_range: float = 1.0) -> pd.DataFrame:
    """
    Public entry point called from nlp_analyzer.enrich_dataframe().

    Loads the model (cached) then runs inference on the full DataFrame.

    Parameters
    ----------
    df              : pd.DataFrame
    progress_bar    : st.progress object or None
    progress_offset : float
    progress_range  : float

    Returns
    -------
    pd.DataFrame with is_sarcastic and sarcasm_confidencs columns added.
    If the model fails to load, returns df with both columns set to None.
    """
    pipe = load_sarcasm_model()

    if pipe is None:
        df = df.copy()
        df["is_sarcastic"]  = None
        df["sarcasm_confidence"] = None
        return df

    return run_sarcasm_analysis(
        df, pipe,
        progress_bar=progress_bar,
        progress_offset=progress_offset,
        progress_range=progress_range
    )
