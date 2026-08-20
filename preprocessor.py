import re
import warnings
import pandas as pd


# ---------------------------------------------------------------------------
# All known WhatsApp timestamp formats, grouped by platform / locale
#
# ANDROID formats  (separator is " - " after the timestamp)
#   DD/MM/YY,  H:MM am/pm  – India / most of Asia  e.g. "07/08/24, 6:23 pm -"
#   DD/MM/YYYY, H:MM am/pm – same with 4-digit year  e.g. "07/08/2024, 6:23 pm -"
#   M/D/YY,  H:MM AM/PM   – US locale               e.g. "9/12/24, 9:19 PM -"
#   DD/MM/YY, HH:MM        – 24-hour Android         e.g. "07/08/24, 18:23 -"
#   D.M.YYYY, HH:MM        – German / European       e.g. "7.8.2024, 18:23 -"
#   YYYY/MM/DD, HH:MM      – some East-Asian locales e.g. "2024/08/07, 18:23 -"
#
# iOS formats  (timestamp is wrapped in square brackets, seconds included)
#   [DD/MM/YYYY, HH:MM:SS AM/PM] Name: message
#   [M/D/YYYY, H:MM:SS AM/PM] Name: message
#   [D.M.YYYY, HH:MM:SS] Name: message
# ---------------------------------------------------------------------------

# ── 1. Detect which broad format the file uses ──────────────────────────────

def _detect_format(data: str) -> str:
    """Return 'ios' or 'android' based on the WhatsApp export format."""

    ios_line = re.compile(
        r"^\["
        r"(?:"
            r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}"
            r"|"
            r"\d{4}[/.]\d{1,2}[/.]\d{1,2}"
        r")"
        r",\s\d{1,2}:\d{2}"
        r"(?::\d{2})?"
        r"(?:\s?[AaPp][Mm]\.?)?"
        r"\]",
        re.MULTILINE
    )

    if ios_line.search(data):
        return "ios"

    return "android"


# ── 2. Normalise the raw text so one regex can handle everything ────────────

def _normalise(data: str) -> str:

    replacements = {
        "\u202f": " ",  # narrow no-break space
        "\u00a0": " ",  # non-breaking space
        "\u2009": " ",  # thin space
        "\u200b": "",   # zero-width space
        "\u2060": "",   # word joiner
        "\ufeff": "",   # BOM
    }

    for old, new in replacements.items():
        data = data.replace(old, new)

    return data



# ── 3. Android parsing ──────────────────────────────────────────────────────
#
# Timestamp token: date-part , time-part  -
#
# date-part  = one of:
#   \d{1,2}/\d{1,2}/\d{2,4}   →  D/M/YY  or  M/D/YY  (slash-separated)
#   \d{1,2}\.\d{1,2}\.\d{2,4} →  D.M.YYYY (dot-separated, European)
#   \d{4}/\d{1,2}/\d{1,2}     →  YYYY/MM/DD (East-Asian)
#
# time-part  = \d{1,2}:\d{2}(:\d{2})? followed by optional AM/PM
#
_AND_TIMESTAMP = (
    r"(?:"
        r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}"
        r"|"
        r"\d{4}[/.]\d{1,2}[/.]\d{1,2}"
    r")"
    r",\s"
    r"\d{1,2}:\d{2}"
    r"(?::\d{2})?"
    r"(?:\s?[AaPp][Mm]\.?)?"
)

_AND_FULL_PATTERN = _AND_TIMESTAMP + r"\s-\s"

_AND_DATE_ONLY = r"(?m)^" + _AND_TIMESTAMP


def _parse_android(data: str) -> pd.DataFrame:

    messages = re.split(_AND_FULL_PATTERN, data)[1:]
    dates = re.findall(_AND_DATE_ONLY, data)

    if not dates:
        return pd.DataFrame()

    min_length = min(len(messages), len(dates))

    df = pd.DataFrame({
        "user_message": messages[:min_length],
        "message_date": dates[:min_length]
    })

    df["message_date"] = _smart_parse(df["message_date"])

    return df[df["message_date"].notnull()].reset_index(drop=True)



# ── 4. iOS parsing ──────────────────────────────────────────────────────────
#
# Example line:
#   [08/07/2024, 6:23:45 PM] Arti Jangid: hello
#
_IOS_TIMESTAMP = (
    r"\["
    r"(?:"
        r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}"
        r"|"
        r"\d{4}[/.]\d{1,2}[/.]\d{1,2}"
    r")"
    r",\s"
    r"\d{1,2}:\d{2}"
    r"(?::\d{2})?"
    r"(?:\s?[AaPp][Mm]\.?)?"
    r"\]"
)

_IOS_FULL_PATTERN = _IOS_TIMESTAMP + r"\s"

_IOS_DATE_ONLY = (
    r"(?m)^\["
    r"("
        r"(?:"
            r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}"
            r"|"
            r"\d{4}[/.]\d{1,2}[/.]\d{1,2}"
        r")"
        r",\s"
        r"\d{1,2}:\d{2}"
        r"(?::\d{2})?"
        r"(?:\s?[AaPp][Mm]\.?)?"
    r")"
    r"\]"
)

def _parse_ios(data: str) -> pd.DataFrame:

    messages = re.split(_IOS_FULL_PATTERN, data)[1:]
    dates = re.findall(_IOS_DATE_ONLY, data)

    if not dates:
        return pd.DataFrame()

    min_length = min(len(messages), len(dates))

    df = pd.DataFrame({
        "user_message": messages[:min_length],
        "message_date": dates[:min_length]
    })

    df["message_date"] = _smart_parse(df["message_date"])

    return df[df["message_date"].notnull()].reset_index(drop=True)



# ── 5. Smart date parser ────────────────────────────────────────────────────
#
# WhatsApp can use either D/M/Y or M/D/Y depending on the phone's locale.
# We try dayfirst=True (most of the world) first.  If that produces too many
# NaTs we fall back to dayfirst=False (US locale).
#
def _smart_parse(date_series: pd.Series) -> pd.Series:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        parsed_day_first = pd.to_datetime(
            date_series,
            dayfirst=True,
            errors="coerce"
        )

        parsed_us = pd.to_datetime(
            date_series,
            dayfirst=False,
            errors="coerce"
        )

    null_day_first = parsed_day_first.isna().sum()
    null_us = parsed_us.isna().sum()

    return (
        parsed_day_first
        if null_day_first <= null_us
        else parsed_us
    )


# ── 6. Split user / message ─────────────────────────────────────────────────

def _extract_users_messages(df: pd.DataFrame):

    users = []
    messages = []

    for raw in df["user_message"]:

        if ": " in raw:
            user, message = raw.split(": ", 1)

            users.append(user)
            messages.append(message)

        else:
            users.append("group_notification")
            messages.append(raw)

    df["user"] = users
    df["message"] = messages

    df.drop(columns=["user_message"], inplace=True)

    return df


# ── 7. Public entry-point ───────────────────────────────────────────────────

def preprocess(data: str) -> pd.DataFrame:
    """
    Parse a WhatsApp chat export (any platform / locale) and return a
    tidy DataFrame with columns:
        date, user, message, only_date, year, month_num, month,
        day, day_name, hour, minute, period
    """
    data = _normalise(data)
    fmt  = _detect_format(data)

    df = _parse_ios(data) if fmt == "ios" else _parse_android(data)

    if df.empty:
        raise ValueError(
            "Could not parse the chat file. "
            "The format may be unsupported or the file may be corrupted."
        )

    df.rename(columns={'message_date': 'date'}, inplace=True)
    df = _extract_users_messages(df)

    # ── Date/time feature columns ───────────────────────────────────────────
    df['only_date'] = df['date'].dt.date
    df['year']      = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month']     = df['date'].dt.month_name()
    df['day']       = df['date'].dt.day
    df['day_name']  = df['date'].dt.day_name()
    df['hour']      = df['date'].dt.hour
    df['minute']    = df['date'].dt.minute

    df['period'] = df['hour'].apply(
        lambda h: f"{h:02d}-{(h + 1) % 24:02d}"
    )

    return df
