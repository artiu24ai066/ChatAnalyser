import re
import pandas as pd

def preprocess(data):
    # normalize unicode spaces that appear in some WhatsApp exports
    data = data.replace('\u202f', ' ').replace('\u00A0', ' ')

    # pattern matches timestamps like: 07/08/24, 6:23 pm -  or 07/08/2024, 18:23 -
    # The (?m) flag makes ^ match start of each line, preventing false matches
    # inside URLs or message content that contain date-like patterns (e.g. 1/22/26
    # inside an Instagram URL like /reel/DTPI_PFgVDv/?igsh=...)
    timestamp_pattern = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[apAP][Mm]\.?)?\s-\s"

    messages = re.split(timestamp_pattern, data)[1:]

    # Use re.MULTILINE so ^ anchors to start of line, not just start of string.
    # This ensures only timestamps that appear at the beginning of a line are
    # captured as dates — not date-like patterns inside URLs or message text.
    dates = re.findall(
        r"(?m)^\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[apAP][Mm]\.?)?",
        data
    )

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # parse dates robustly (day-first), allow both 12h and 24h formats
    df['message_date'] = pd.to_datetime(df['message_date'], dayfirst=True, errors='coerce')
    # drop rows that failed to parse
    df = df[df['message_date'].notnull()]

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df['hour']:
        start = f"{hour:02d}"
        end = f"{(hour + 1) % 24:02d}"
        period.append(f"{start}-{end}")

    df['period'] = period

    return df
