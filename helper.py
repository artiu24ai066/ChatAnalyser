from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    def extract_emojis(message):
        if hasattr(emoji, 'emoji_list'):
            return [item['emoji'] for item in emoji.emoji_list(message)]
        if hasattr(emoji, 'EMOJI_DATA'):
            return [c for c in message if c in emoji.EMOJI_DATA]
        return []

    emojis = []
    for message in df['message']:
        emojis.extend(extract_emojis(message))

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap



def emoji_emotion_analysis(selected_user, df):
    """
    Connect the extracted_emojis column (Phase 1) with the emotion column (Phase 2).

    For every message that has BOTH emojis AND a predicted emotion, we record
    each (emoji, emotion) pair. This lets us answer questions like:
      - Which emojis appear most often in "Joy" messages?
      - Which emojis appear most often in "Anger" messages?
      - What is the most common emotion for ❤️? For 😭? For 😂?

    HOW IT WORKS
    ------------
    A message like "bahut maza aaya 😂😂" predicted as "Joy" contributes
    two rows to the result:
        (😂, Joy)
        (😂, Joy)

    We count how many times each (emoji, emotion) pair appears, then
    build a clean summary table.

    IMPORTANT: This is OBSERVED association, not absolute truth.
    The same emoji can appear in different emotions depending on context.
    Example: 😭 can appear in both Sadness AND Surprise.

    Parameters
    ----------
    selected_user : str   — "Overall" or a specific user name
    df            : pd.DataFrame — enriched DataFrame (must have
                    'extracted_emojis' and 'emotion' columns)

    Returns
    -------
    tuple of three DataFrames:

    1. pair_df — every (emoji, emotion) pair with its count, sorted by count desc
       Columns: emoji, emotion, count

    2. pivot_df — pivot table: rows=emoji, columns=emotion, values=count
       Useful for a heatmap. Only top N emojis by total frequency.

    3. top_per_emotion_df — for each emotion, the top 5 most common emojis
       Columns: emotion, emoji, count
    """
    # Filter for selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # We need rows that have BOTH an emotion prediction AND at least one emoji.
    # Rows without emotion (media, notifications) are already None — drop them.
    # Rows with empty emoji list also contribute nothing useful — drop them.
    df = df[df['emotion'].notna()]                        # must have emotion
    df = df[df['extracted_emojis'].apply(                 # must have ≥1 emoji
        lambda x: isinstance(x, list) and len(x) > 0
    )]

    if df.empty:
        # Return three empty DataFrames with correct columns so app.py
        # can handle the empty case cleanly without crashing.
        empty_pair     = pd.DataFrame(columns=['emoji', 'emotion', 'count'])
        empty_pivot    = pd.DataFrame()
        empty_top      = pd.DataFrame(columns=['emotion', 'emoji', 'count'])
        return empty_pair, empty_pivot, empty_top

    # Build a flat list of (emoji, emotion) pairs.
    # One message can have multiple emojis — each gets its own row.
    pairs = []
    for _, row in df.iterrows():
        for em in row['extracted_emojis']:
            pairs.append({'emoji': em, 'emotion': row['emotion']})

    pair_df = pd.DataFrame(pairs)

    # Count how many times each (emoji, emotion) pair appears
    pair_df = (
        pair_df
        .groupby(['emoji', 'emotion'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )

    # ── Pivot table for heatmap ───────────────────────────────────────────────
    # Limit to the top 15 most frequent emojis to keep the heatmap readable.
    # More than 15 rows makes the chart cramped.
    top_emojis = (
        pair_df.groupby('emoji')['count']
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
        .tolist()
    )

    pivot_data = pair_df[pair_df['emoji'].isin(top_emojis)]

    if pivot_data.empty:
        pivot_df = pd.DataFrame()
    else:
        pivot_df = pivot_data.pivot_table(
            index='emoji',
            columns='emotion',
            values='count',
            fill_value=0
        )
        # Sort rows by total emoji frequency (most used emoji at top)
        pivot_df = pivot_df.loc[top_emojis]

    # ── Top 5 emojis per emotion ──────────────────────────────────────────────
    top_per_emotion_df = (
        pair_df
        .sort_values('count', ascending=False)
        .groupby('emotion')
        .head(5)                     # top 5 emojis for each emotion
        .reset_index(drop=True)
        [['emotion', 'emoji', 'count']]
        .sort_values(['emotion', 'count'], ascending=[True, False])
        .reset_index(drop=True)
    )

    return pair_df, pivot_df, top_per_emotion_df


def sentiment_by_user(selected_user, df, min_messages=5):
    """
    Calculate sentiment distribution (%) for each user.

    When selected_user == "Overall":
        Returns a pivot table with one row per user, columns = Positive/Neutral/Negative.
        Each cell = percentage of that user's messages with that sentiment.
        Users with fewer than min_messages analyzed messages are excluded (too noisy).

    When selected_user is a specific user:
        Returns the same pivot table but with only that user's row.
        This gives the individual user's own sentiment breakdown.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame  (must have 'user', 'sentiment' columns)
    min_messages  : int  — minimum analyzed messages needed to include a user (default 5)

    Returns
    -------
    pivot : pd.DataFrame
        Rows = users, Columns = ["Positive", "Neutral", "Negative"]
        Values = percentage (0–100), rounded to 1 decimal place.
        Empty DataFrame if no valid data.

    counts : pd.DataFrame
        Same shape as pivot but raw message counts instead of percentages.
        Useful to show alongside the chart so the user knows sample sizes.
    """
    # Filter for selected user if not Overall
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Drop rows without sentiment (media, notifications, etc.)
    df = df[df['sentiment'].notna()]

    # Exclude group_notification user
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Count messages per (user, sentiment) combination
    counts_long = (
        df.groupby(['user', 'sentiment'])
        .size()
        .reset_index(name='count')
    )

    # Total analyzed messages per user
    user_totals = (
        counts_long.groupby('user')['count']
        .sum()
        .reset_index(name='total')
    )

    # Filter out users below the minimum threshold
    valid_users = user_totals[user_totals['total'] >= min_messages]['user'].tolist()
    counts_long = counts_long[counts_long['user'].isin(valid_users)]

    if counts_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Build raw counts pivot (users × sentiments)
    counts = counts_long.pivot_table(
        index='user',
        columns='sentiment',
        values='count',
        fill_value=0
    )

    # Ensure all three sentiment columns exist even if a sentiment never appears
    for col in ['Positive', 'Neutral', 'Negative']:
        if col not in counts.columns:
            counts[col] = 0

    # Keep only the three standard columns in a logical order
    counts = counts[['Positive', 'Neutral', 'Negative']]

    # Convert raw counts → percentages
    row_totals = counts.sum(axis=1)
    pivot = counts.div(row_totals, axis=0).multiply(100).round(1)

    # Sort users by their Positive % descending (most positive user at top)
    pivot = pivot.sort_values('Positive', ascending=False)
    counts = counts.loc[pivot.index]   # keep same row order

    return pivot, counts


def emotion_by_user(selected_user, df, min_messages=5):
    """
    Calculate emotion distribution (%) for each user.

    Same structure as sentiment_by_user() but for the 7 emotion classes.

    When selected_user == "Overall":
        Returns a pivot with one row per user, columns = each emotion present in data.

    When selected_user is a specific user:
        Returns only that user's emotion breakdown.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame  (must have 'user', 'emotion' columns)
    min_messages  : int  — minimum analyzed messages to include a user (default 5)

    Returns
    -------
    pivot : pd.DataFrame
        Rows = users, Columns = emotion labels present in data.
        Values = percentage (0–100), rounded to 1 decimal place.

    counts : pd.DataFrame
        Same shape as pivot but raw message counts.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['emotion'].notna()]
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts_long = (
        df.groupby(['user', 'emotion'])
        .size()
        .reset_index(name='count')
    )

    user_totals = (
        counts_long.groupby('user')['count']
        .sum()
        .reset_index(name='total')
    )

    valid_users = user_totals[user_totals['total'] >= min_messages]['user'].tolist()
    counts_long = counts_long[counts_long['user'].isin(valid_users)]

    if counts_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    counts = counts_long.pivot_table(
        index='user',
        columns='emotion',
        values='count',
        fill_value=0
    )

    row_totals = counts.sum(axis=1)
    pivot = counts.div(row_totals, axis=0).multiply(100).round(1)

    # Sort users by total message count descending (most active user first)
    user_order = user_totals.set_index('user')['total']
    user_order = user_order[user_order.index.isin(pivot.index)].sort_values(ascending=False)
    pivot  = pivot.loc[user_order.index]
    counts = counts.loc[user_order.index]

    return pivot, counts


def sentiment_over_time(selected_user, df):
    """
    Calculate monthly sentiment distribution (%) over time.

    Groups messages by year+month, then for each month calculates what
    percentage of that month's messages were Positive / Neutral / Negative.

    This lets you see how the overall emotional tone of the chat changed
    across months — e.g. was March more positive than December?

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame  (must have 'user', 'sentiment',
                                   'year', 'month_num', 'month' columns)

    Returns
    -------
    result : pd.DataFrame
        Columns: time_label (str e.g. "Jan-2024"),
                 Positive (%), Neutral (%), Negative (%),
                 total (int — total analyzed messages that month)
        Sorted chronologically (oldest month first).
        Empty DataFrame if no valid data.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['sentiment'].notna()]
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame()

    # Group by year + month_num + month name to get correct chronological order
    grouped = (
        df.groupby(['year', 'month_num', 'month', 'sentiment'])
        .size()
        .reset_index(name='count')
    )

    # Pivot so each sentiment becomes a column
    pivot = grouped.pivot_table(
        index=['year', 'month_num', 'month'],
        columns='sentiment',
        values='count',
        fill_value=0
    ).reset_index()

    # Ensure all three sentiment columns exist
    for col in ['Positive', 'Neutral', 'Negative']:
        if col not in pivot.columns:
            pivot[col] = 0

    # Total messages analyzed per month (for context)
    pivot['total'] = pivot['Positive'] + pivot['Neutral'] + pivot['Negative']

    # Convert raw counts → percentages
    for col in ['Positive', 'Neutral', 'Negative']:
        pivot[f'{col}_pct'] = (pivot[col] / pivot['total'] * 100).round(1)

    # Sort chronologically (year asc, then month_num asc)
    pivot = pivot.sort_values(['year', 'month_num']).reset_index(drop=True)

    # Build a human-readable time label: "Jan-2024"
    pivot['time_label'] = pivot['month'].str[:3] + '-' + pivot['year'].astype(str)

    # Return clean output with only the columns app.py needs
    result = pivot[['time_label', 'Positive_pct', 'Neutral_pct', 'Negative_pct', 'total']].copy()
    result = result.rename(columns={
        'Positive_pct': 'Positive',
        'Neutral_pct':  'Neutral',
        'Negative_pct': 'Negative',
    })

    return result


def emotion_over_time(selected_user, df, top_n=5):
    """
    Calculate monthly emotion distribution (%) over time.

    Shows the top_n most frequent emotions, but percentages are calculated
    against ALL analyzed messages that month — not just the top-5 subset.

    This is the scientifically correct approach:
      - Dropping low-frequency emotions from the denominator would inflate
        the percentages of displayed emotions.
      - Example: if Joy=40, Neutral=30, Sadness=10, Anger=10, Love=5, Fear=5
        and we show top-5 (dropping Fear), the denominator should still be 100
        (all messages), not 95 (top-5 only). Otherwise Joy would show as
        40/95=42% instead of the correct 40/100=40%.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame  (must have 'user', 'emotion',
                                   'year', 'month_num', 'month' columns)
    top_n         : int  — how many emotions to display (default 5)

    Returns
    -------
    result : pd.DataFrame
        Columns: time_label (str), plus one column per top emotion (%).
        Percentages are relative to ALL analyzed messages that month.
        Sorted chronologically. Empty DataFrame if no valid data.

    top_emotions : list of str
        The emotion labels selected for display.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['emotion'].notna()]
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame(), []

    # Find top_n most frequent emotions overall (for display selection only)
    top_emotions = (
        df['emotion'].value_counts()
        .head(top_n)
        .index
        .tolist()
    )

    # ── Step 1: Count ALL emotions per month (for correct denominator) ────────
    all_grouped = (
        df.groupby(['year', 'month_num', 'month'])
        .size()
        .reset_index(name='total_all')
    )

    # ── Step 2: Count only the top-N emotions per month ───────────────────────
    top_grouped = (
        df[df['emotion'].isin(top_emotions)]
        .groupby(['year', 'month_num', 'month', 'emotion'])
        .size()
        .reset_index(name='count')
    )

    pivot = top_grouped.pivot_table(
        index=['year', 'month_num', 'month'],
        columns='emotion',
        values='count',
        fill_value=0
    ).reset_index()

    # Ensure all top_emotions columns exist even if absent in some months
    for emo in top_emotions:
        if emo not in pivot.columns:
            pivot[emo] = 0

    # ── Step 3: Merge correct total (all messages) as denominator ─────────────
    pivot = pivot.merge(all_grouped, on=['year', 'month_num', 'month'], how='left')

    # ── Step 4: Calculate % against all analyzed messages ─────────────────────
    for emo in top_emotions:
        pivot[f'{emo}_pct'] = (pivot[emo] / pivot['total_all'] * 100).round(1)

    pivot = pivot.sort_values(['year', 'month_num']).reset_index(drop=True)
    pivot['time_label'] = pivot['month'].str[:3] + '-' + pivot['year'].astype(str)

    pct_cols = [f'{e}_pct' for e in top_emotions]
    result = pivot[['time_label'] + pct_cols].copy()
    result = result.rename(columns={f'{e}_pct': e for e in top_emotions})

    return result, top_emotions


def sentiment_trend(selected_user, df):
    """
    Calculate a numerical sentiment trend score per month.

    Maps each sentiment label to a number:
        Positive → +1
        Neutral  →  0
        Negative → -1

    Then takes the average across all messages in each month.

    Result ranges from -1.0 (all messages Negative) to +1.0 (all Positive).
    A score near 0 means a roughly balanced or mostly Neutral month.

    This is a STATISTICAL score, not a psychological measurement.
    It represents the average predicted sentiment polarity for that month.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame

    Returns
    -------
    trend : pd.DataFrame
        Columns: time_label (str), trend_score (float, -1 to +1),
                 total (int — messages analyzed that month)
        Sorted chronologically.
        Empty DataFrame if no valid data.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['sentiment'].notna()]
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame()

    # Map sentiment labels to numeric scores
    score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df = df.copy()
    df['sentiment_score_num'] = df['sentiment'].map(score_map)

    # Average score per month
    trend = (
        df.groupby(['year', 'month_num', 'month'])
        .agg(
            trend_score=('sentiment_score_num', 'mean'),
            total=('sentiment_score_num', 'count')
        )
        .reset_index()
    )

    trend['trend_score'] = trend['trend_score'].round(3)
    trend = trend.sort_values(['year', 'month_num']).reset_index(drop=True)
    trend['time_label'] = trend['month'].str[:3] + '-' + trend['year'].astype(str)

    return trend[['time_label', 'trend_score', 'total']]


def top_positive_negative_messages(selected_user, df, n=10):
    """
    Return the top-N most confidently Positive and top-N most confidently
    Negative messages, ranked by the sentiment model's confidence score.

    The sentiment_score column (0.0–1.0) represents how certain the model
    was about its label — NOT emotional intensity.
    Example: score=0.98 for Positive means 98% model confidence.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame  (must have sentiment, sentiment_score columns)
    n             : int  — how many messages per category (default 10)

    Returns
    -------
    top_positive : pd.DataFrame
        Columns: user, date, message, sentiment, sentiment_score
        Sorted by sentiment_score descending.

    top_negative : pd.DataFrame
        Same structure for Negative messages.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Only rows with a valid sentiment prediction
    df = df[df['sentiment'].notna()].copy()
    df = df[df['user'] != 'group_notification']

    empty = pd.DataFrame(columns=['user', 'date', 'message', 'sentiment', 'sentiment_score'])
    if df.empty:
        return empty.copy(), empty.copy()

    display_cols = ['user', 'date', 'message', 'sentiment', 'sentiment_score']
    df = df[display_cols].copy()
    df['sentiment_score'] = df['sentiment_score'].astype(float).round(4)

    top_positive = (
        df[df['sentiment'] == 'Positive']
        .sort_values('sentiment_score', ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    top_negative = (
        df[df['sentiment'] == 'Negative']
        .sort_values('sentiment_score', ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    return top_positive, top_negative


def sarcasm_distribution(selected_user, df):
    """
    Calculate overall sarcasm distribution for the selected user.

    Returns counts and percentages of sarcastic vs non-sarcastic messages.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame  (must have 'is_sarcastic' column)

    Returns
    -------
    result : pd.DataFrame
        Columns: label (str), count (int), percentage (float)
        Two rows: 'Sarcastic' and 'Not Sarcastic'
        Empty DataFrame if no valid data.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['is_sarcastic'].notna()]
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame(columns=['label', 'count', 'percentage'])

    total      = len(df)
    sarc_count = int(df['is_sarcastic'].sum())
    non_count  = total - sarc_count

    result = pd.DataFrame({
        'label':      ['Sarcastic', 'Not Sarcastic'],
        'count':      [sarc_count, non_count],
        'percentage': [
            round(sarc_count / total * 100, 1),
            round(non_count  / total * 100, 1)
        ]
    })
    return result


def sarcasm_by_user(selected_user, df, min_messages=5):
    """
    Calculate sarcasm rate (%) per user.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame
    min_messages  : int  — minimum analyzed messages to include a user

    Returns
    -------
    result : pd.DataFrame
        Columns: user, sarcastic_count, total_analyzed, sarcasm_rate (%)
        Sorted by sarcasm_rate descending.
        Empty DataFrame if no valid data.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['is_sarcastic'].notna()]
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame(columns=['user', 'sarcastic_count', 'total_analyzed', 'sarcasm_rate'])

    grouped = (
        df.groupby('user')
        .agg(
            sarcastic_count=('is_sarcastic', 'sum'),
            total_analyzed=('is_sarcastic', 'count')
        )
        .reset_index()
    )

    # Filter out users below the minimum threshold
    grouped = grouped[grouped['total_analyzed'] >= min_messages]

    if grouped.empty:
        return pd.DataFrame(columns=['user', 'sarcastic_count', 'total_analyzed', 'sarcasm_rate'])

    grouped['sarcasm_rate'] = (
        grouped['sarcastic_count'] / grouped['total_analyzed'] * 100
    ).round(1)

    grouped['sarcastic_count'] = grouped['sarcastic_count'].astype(int)

    return grouped.sort_values('sarcasm_rate', ascending=False).reset_index(drop=True)


def sarcasm_over_time(selected_user, df):
    """
    Calculate monthly sarcasm rate (%) over time.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame  (must have 'is_sarcastic', 'year',
                                   'month_num', 'month' columns)

    Returns
    -------
    result : pd.DataFrame
        Columns: time_label (str), sarcasm_rate (%), total (int)
        Sorted chronologically.
        Empty DataFrame if no valid data.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['is_sarcastic'].notna()]
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame()

    trend = (
        df.groupby(['year', 'month_num', 'month'])
        .agg(
            sarcastic_count=('is_sarcastic', 'sum'),
            total=('is_sarcastic', 'count')
        )
        .reset_index()
    )

    trend['sarcasm_rate'] = (trend['sarcastic_count'] / trend['total'] * 100).round(1)
    trend = trend.sort_values(['year', 'month_num']).reset_index(drop=True)
    trend['time_label'] = trend['month'].str[:3] + '-' + trend['year'].astype(str)

    return trend[['time_label', 'sarcasm_rate', 'total']]


def top_sarcastic_messages(selected_user, df, n=10):
    """
    Return the top-N messages most confidently predicted as sarcastic,
    along with their sentiment and emotion context.

    Parameters
    ----------
    selected_user : str
    df            : pd.DataFrame
    n             : int  — number of messages to return (default 10)

    Returns
    -------
    pd.DataFrame
        Columns: user, date, message, sarcasm_score, sentiment, emotion
        Sorted by sarcasm_score descending.
        Empty DataFrame if no sarcastic messages found.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # is_sarcastic is stored as Python bool objects in an object-dtype column.
    # Using == True can silently fail on object columns — .isin([True]) is reliable.
    df = df[df['is_sarcastic'].isin([True])].copy()
    df = df[df['user'] != 'group_notification']

    if df.empty:
        return pd.DataFrame(
            columns=['user', 'date', 'message', 'sarcasm_score', 'sentiment', 'emotion']
        )

    # Pull available columns — sentiment/emotion may not always be present
    cols = ['user', 'date', 'message', 'sarcasm_score']
    if 'sentiment' in df.columns:
        cols.append('sentiment')
    if 'emotion' in df.columns:
        cols.append('emotion')

    result = (
        df[cols]
        .sort_values('sarcasm_score', ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    result['sarcasm_score'] = result['sarcasm_score'].astype(float).round(4)
    return result
