# ChatScope: WhatsApp Chat Analyser

**ChatScope** is an interactive Streamlit application that transforms a WhatsApp `.txt` export into a comprehensive analytics and NLP dashboard.

It combines **conversation statistics, activity analysis, word and emoji analytics, multilingual sentiment and emotion classification, and message-level sarcasm detection** in one application.

> **Upload → Analyse → Explore your conversation**

---

## Features

### 📊 Conversation Analytics

* Parse Android and iOS WhatsApp `.txt` exports.
* Support common day-first, US-style, European, and ISO-like date formats.
* Analyse the complete conversation or a specific participant.
* View message, word, media, and link totals.
* Identify busiest users, days, months, and time periods.

### 📅 Activity Analysis

* Monthly and daily activity timelines.
* Day-of-week and hourly activity analysis.
* Activity heatmaps.
* Participant-level activity comparison.

### 📝 Word & Emoji Analysis

* Generate word clouds.
* Identify the most frequently used words.
* Hinglish stop-word filtering.
* Analyse emoji frequency.
* Explore emoji and emotion co-occurrences.

### ❤️ NLP Analysis

* **Sentiment:** Positive, Neutral, Negative.
* **Emotion:** 11 multilingual emotion categories.
* **Sarcasm:** Sarcastic / Not Sarcastic.
* Confidence scores and message-level predictions.
* Trend, distribution, heatmap, and top-confidence analysis.

---

## 🧠 NLP Models

| Task      | Model                                              | Output                        |
| --------- | -------------------------------------------------- | ----------------------------- |
| Sentiment | `shae2977/xlm-roberta-hinglish-sentiment-analysis` | Positive / Neutral / Negative |
| Emotion   | `tabularisai/multilingual-emotion-classification`  | 11 emotion categories         |
| Sarcasm   | `amaan00z/sarcasm_xlmr`                            | Sarcastic / Not Sarcastic     |

All models use batched inference and Streamlit resource caching.

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **Streamlit** — interactive dashboard
* **Pandas / NumPy** — data processing
* **Matplotlib / Seaborn** — visualisation
* **WordCloud** — word-frequency visualisation
* **Hugging Face Transformers** — NLP inference
* **PyTorch** — transformer model execution
* **Regex** — WhatsApp message parsing

---

## 🔄 Application Workflow

```text
WhatsApp .txt Export
        │
        ▼
┌─────────────────────┐
│   preprocessor.py   │
│                     │
│ Parse messages      │
│ Parse timestamps    │
│ Extract users       │
│ Create time fields  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   nlp_analyzer.py   │
│                     │
│ Emoji extraction    │
│ Sentiment           │
│ Emotion             │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│ sarcasm_analyzer.py  │
│                      │
│ Sarcasm prediction   │
│ Confidence scores    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│       app.py        │
│                     │
│ Statistics          │
│ Charts              │
│ Tables               │
│ NLP Insights        │
└─────────────────────┘
```

---

## 📁 Project Structure

```text
.
├── app.py                       # Streamlit UI and visualisations
├── helper.py                    # Statistics and analysis helpers
├── nlp_analyzer.py              # Sentiment, emotion, emoji analysis
├── preprocessor.py              # WhatsApp parsing and preprocessing
├── sarcasm_analyzer.py          # Sarcasm model and inference
├── requirements.txt             # Python dependencies
├── stop_hinglish.txt            # Hinglish stop-word list
│
├── WhatsApp Chat*.txt           # Example WhatsApp exports
│
└── evaluate/
    ├── ...                      # Evaluation scripts
    └── ...                      # Evaluation documentation
```

---

## ⚙️ Requirements

* Python **3.9 or newer**
* Internet access for the first model download
* Sufficient RAM and storage for transformer models

The three NLP models may require significant memory during analysis.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows CMD**

```bat
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

### Usage

1. Upload a WhatsApp `.txt` export.
2. Select **Overall** or a participant.
3. Click **Show Analysis**.
4. Explore statistics, activity, words, emojis, sentiment, emotion, and sarcasm.

> The first analysis may take longer because the transformer models need to be downloaded and loaded.

---

## 📱 Supported WhatsApp Formats

Example Android-style export:

```text
07/08/24, 6:23 pm - Alice: Hello
```

Example iOS-style export:

```text
[08/07/2024, 6:23:45 PM] Alice: Hello
```

The parser supports:

* Android exports
* iOS exports
* Multiple timestamp formats
* Multi-line messages
* Group/system notifications

Rows without a valid `User: Message` separator are treated as `group_notification` rows.

If parsing fails, verify that the file is a standard WhatsApp `.txt` export rather than a copied transcript, HTML, or JSON export.

---

# 🧩 Core Modules

## `preprocessor.py`

Public entry point:

```python
preprocess(data)
```

Responsibilities:

* Normalise invisible whitespace.
* Detect Android/iOS formatting.
* Parse timestamps.
* Separate users and messages.
* Preserve multi-line messages.
* Identify system/group notifications.
* Generate time-derived features.

Generated columns include:

```text
date
only_date
year
month_num
month
day
day_name
hour
minute
period
```

Malformed or unsupported input raises `ValueError`.

---

## `helper.py`

Contains non-model analysis functions used by the dashboard.

It handles:

* Basic statistics
* Activity timelines
* Busy-user summaries
* Word clouds
* Common-word analysis
* Emoji counts
* Sentiment/emotion summaries
* Cross-analysis tables
* Trends
* Confidence analysis

`stop_hinglish.txt` is used to remove common Hinglish words from word analysis.

---

## `nlp_analyzer.py`

Enriches the preprocessed DataFrame with NLP information.

### Sentiment

Uses:

```text
shae2977/xlm-roberta-hinglish-sentiment-analysis
```

Outputs:

```text
Negative
Neutral
Positive
```

### Emotion

Uses:

```text
tabularisai/multilingual-emotion-classification
```

Outputs 11 multilingual emotion categories.

### Emoji Analysis

Extracts emojis from conversational messages for frequency and co-occurrence analysis.

### Performance

* Batched inference.
* Non-conversational messages are skipped.
* Missing predictions are stored as null values.
* Models use `st.cache_resource`.
* Enriched data is cached using an MD5 hash of the uploaded file.

---

## `sarcasm_analyzer.py`

Uses:

```text
amaan00z/sarcasm_xlmr
```

Adds:

```text
is_sarcastic
sarcasm_confidence
```

Model labels are mapped as:

```text
LABEL_0 → Not Sarcastic
LABEL_1 → Sarcastic
```

Sarcasm inference is performed in batches and the model is cached.

---

# 📊 Model Evaluation

The `evaluate/` directory provides a separate evaluation subsystem for analysing model performance independently from the Streamlit dashboard.

It can be used to examine:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report
* Prediction confidence
* Sample-level predictions

Example result format:

```text
Sarcasm Detection
-----------------
Accuracy  : XX.XX%
Precision : XX.XX%
Recall    : XX.XX%
F1 Score  : XX.XX%
```

> Replace the placeholder values with the final evaluation results from the project.

---

# 🔐 Privacy

ChatScope processes the uploaded chat locally through the running application.

The uploaded file is:

* Decoded in memory.
* Parsed in memory.
* Enriched in memory.
* Not written to a project file by the application.

Transformer models are downloaded from Hugging Face during first use.

If deployed to a shared or third-party server, users should understand its storage, logging, access, and data-retention policies before uploading private conversations.

**Do not upload sensitive conversations to untrusted deployments.**

---

# ⚡ Performance & Caching

ChatScope uses Streamlit caching to reduce repeated computation.

### Model Resources

```python
st.cache_resource
```

is used to cache transformer models.

### Enriched Data

The enriched DataFrame is cached using an MD5 hash of the uploaded file.

### Batched Inference

Sentiment, emotion, and sarcasm predictions are performed in batches instead of processing messages individually.

This improves performance for larger conversations and avoids unnecessary repeated model loading.

---

# 🔍 Interpretation & Limitations

ChatScope predictions are **statistical model outputs, not ground truth**.

### Confidence Scores

Confidence indicates model certainty. It does not represent emotional or sarcasm intensity.

### Sarcasm Context

Sarcasm often depends on:

* Previous messages
* Conversation history
* Tone
* Shared context
* Cultural knowledge

Because ChatScope performs message-level prediction, contextual sarcasm can be missed.

### Informal Language

Performance may be affected by:

* Hinglish
* Code-switching
* Spelling variations
* Abbreviations
* Slang
* Short messages
* Emojis

### Excluded Messages

Non-conversational rows such as system messages, media-only messages, links, and deleted-message notices are excluded from NLP inference.

### Large Chats

Very large exports can require substantial RAM, CPU time, and inference time.

### Date Ambiguity

For ambiguous timestamps, the parser selects the interpretation resulting in fewer invalid timestamps. This may still differ from the sender's local date convention.

---

# 🧪 Development Checks

Check Python syntax with:

```bash
python -m py_compile app.py helper.py nlp_analyzer.py preprocessor.py sarcasm_analyzer.py
```

The main application currently does not include a dedicated automated test suite.

The `evaluate/` directory is maintained separately for model evaluation.

---

# 🚧 Future Improvements

* Conversation-context-aware sarcasm detection
* Better Hinglish-specific preprocessing
* GPU acceleration
* Additional WhatsApp export formats
* Automated unit and integration tests
* More extensive model benchmarking
* Improved large-chat handling
* Conversation/network visualisations
* Exportable analysis reports

---

# 🎯 Project Highlights

ChatScope demonstrates the integration of:

```text
Data Processing
      +
NLP
      +
Transformer Models
      +
Multilingual Analysis
      +
Sarcasm Detection
      +
Data Visualisation
      +
Interactive Web Application
```

The project goes beyond basic WhatsApp statistics by combining **conversation analytics with multilingual NLP, emotion analysis, emoji analysis, and sarcasm detection** in one interactive dashboard.

---

## 👩‍💻 Author

**[Arti Jangid]**

AI / Machine Learning / NLP Project


---

## ⭐ Acknowledgements

This project uses pretrained transformer models from the Hugging Face ecosystem:

* `shae2977/xlm-roberta-hinglish-sentiment-analysis`
* `tabularisai/multilingual-emotion-classification`
* `amaan00z/sarcasm_xlmr`
