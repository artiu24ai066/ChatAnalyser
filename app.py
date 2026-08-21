import streamlit as st
import preprocessor, helper, nlp_analyzer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import hashlib

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d7dfdc",
    "axes.labelcolor": "#697583",
    "axes.titlecolor": "#1d2733",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e8eeeb",
    "grid.linewidth": 0.8,
    "xtick.color": "#697583",
    "ytick.color": "#697583",
    "text.color": "#1d2733",
    "font.family": "sans-serif",
    "font.size": 10,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})

st.set_page_config(
    page_title="ChatScope | WhatsApp Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #1d2733;
    --muted: #697583;
    --paper: #f7f8f5;
    --panel: #ffffff;
    --line: #e4e8e4;
    --mint: #168f72;
    --mint-soft: #e6f4ef;
    --coral: #e56c52;
    --yellow: #f4c95d;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
[data-testid="stAppViewContainer"] { background: var(--paper); }
[data-testid="stHeader"] { background: rgba(247, 248, 245, 0.82); }
[data-testid="stSidebar"] { background: #172b35; border-right: 0; }
[data-testid="stSidebar"] label { color: #f4f7f4 !important; }
[data-testid="stSidebar"] .stFileUploader { background: rgba(255,255,255,.08); border: 1px dashed rgba(255,255,255,.3); border-radius: 12px; padding: 8px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #203d48; border-color: rgba(255,255,255,.38); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] { color: #f4f7f4 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small { color: #b7c9c5 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background: var(--yellow) !important; color: #172b35 !important; border: 0 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { color: #172b35 !important; }
[data-testid="stSidebar"] section[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { background: #294b57 !important; border-radius: 8px; }
[data-testid="stSidebar"] section[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
[data-testid="stSidebar"] section[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
[data-testid="stSidebar"] section[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] { color: #ffffff !important; opacity: 1 !important; }
[data-testid="stSidebar"] section[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg { fill: #ffffff !important; color: #ffffff !important; }
[data-testid="stSidebar"] section[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button { background: transparent !important; color: #ffffff !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #203d48 !important; border-color: rgba(255,255,255,.38) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] [role="option"],
[data-testid="stSidebar"] [data-baseweb="select"] [aria-selected="true"],
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] > div > div { color: #ffffff !important; opacity: 1 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #f4f7f4 !important; color: #f4f7f4 !important; }
[data-testid="stSidebar"] .stButton > button { background: var(--yellow); color: #172b35; border: 0; font-weight: 700; }
[data-testid="stSidebar"] .stButton > button:hover { background: #f8d879; color: #172b35; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
h1 { font-size: 2.55rem !important; line-height: 1.05 !important; }
h2 { margin-top: 1.8rem !important; }
.brand-mark { color: #f4f7f4; font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em; margin: 4px 0 30px; }
.brand-mark span { color: var(--yellow); }
.sidebar-kicker { color: #9fc8bb; text-transform: uppercase; font-size: .69rem; font-weight: 700; letter-spacing: .14em; margin: 0 0 6px; }
.sidebar-copy { color: #b7c9c5; font-size: .83rem; line-height: 1.5; margin-bottom: 22px; }
.hero { background: #ffffff; border: 1px solid var(--line); border-radius: 18px; padding: 30px 34px 28px; margin: 8px 0 24px; box-shadow: 0 8px 30px rgba(29,39,51,.05); position: relative; overflow: hidden; }
.hero:after { content: ''; position: absolute; width: 180px; height: 180px; border: 22px solid var(--mint-soft); border-radius: 50%; right: -42px; top: -65px; }
.hero-kicker { color: var(--mint); text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; margin-bottom: 10px; }
.hero h1 { margin: 0 0 9px; position: relative; z-index: 1; }
.hero p { color: var(--muted); max-width: 660px; margin: 0; font-size: 1.02rem; position: relative; z-index: 1; }
.hero-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 20px; position: relative; z-index: 1; }
.pill { background: var(--mint-soft); color: #17765f; border-radius: 999px; padding: 6px 11px; font-size: .75rem; font-weight: 700; }
.empty-state { max-width: 820px; margin: 12vh auto 0; text-align: center; padding: 46px 34px; background: #fff; border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 12px 40px rgba(29,39,51,.06); }
.empty-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state h1 { margin: 0 0 12px; }
.empty-state p { color: var(--muted); max-width: 560px; margin: auto; line-height: 1.6; }
.section-intro { color: var(--muted); font-size: .92rem; margin-top: -10px; margin-bottom: 18px; }
.glance-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 34px; }
.glance-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 17px 18px 16px; min-height: 108px; box-shadow: 0 5px 18px rgba(29,39,51,.04); position: relative; overflow: hidden; }
.glance-card:after { content: ''; position: absolute; width: 58px; height: 58px; border: 10px solid var(--mint-soft); border-radius: 50%; right: -19px; bottom: -25px; }
.glance-label { color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.glance-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; line-height: 1.1; margin-top: 13px; position: relative; z-index: 1; }
.timeline-heading { border-top: 1px solid var(--line); padding-top: 24px; }
@media (max-width: 760px) { .glance-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 430px) { .glance-grid { grid-template-columns: 1fr; } }
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 15px 17px; box-shadow: 0 4px 16px rgba(29,39,51,.035); }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; }
[data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; box-shadow: 0 5px 18px rgba(29,39,51,.035); animation: rise-in .5s ease-out both; }
[data-testid="stImage"] { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 10px; box-shadow: 0 5px 18px rgba(29,39,51,.035); animation: rise-in .55s ease-out both; }
[data-testid="stMetric"] { animation: rise-in .45s ease-out both; }
.stPlotlyChart { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; box-shadow: 0 5px 18px rgba(29,39,51,.035); animation: rise-in .55s ease-out both; }
@keyframes rise-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@media (prefers-reduced-motion: reduce) { [data-testid="stDataFrame"], [data-testid="stImage"], [data-testid="stMetric"], .stPlotlyChart { animation: none; } }
.stButton > button { border-radius: 9px; min-height: 2.7rem; font-weight: 700; }
.stProgress > div > div > div { background: var(--mint); }
div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: #fff; }
.footer-note { color: #87928f; font-size: .75rem; text-align: center; margin: 42px 0 12px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="brand-mark">Chat<span>Scope</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-kicker">WhatsApp intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-copy">Turn everyday conversations into patterns, moods, and moments worth noticing.</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-kicker">Start here</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-copy">Upload a WhatsApp chat export in <b>.txt</b> format to begin.</div>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Choose a WhatsApp chat file",
    type=["txt"],
    help="Export your WhatsApp conversation as a .txt file, then upload it here.",
)
if uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">💬</div>
        <div class="hero-kicker">Conversation, made visible</div>
        <h1>Find the story inside your chats.</h1>
        <p>Upload a WhatsApp export from the sidebar to explore activity patterns, language, sentiment, emotion, emojis, and irony in one calm workspace.</p>
        <div class="hero-meta" style="justify-content:center;"><span class="pill">Activity</span><span class="pill">Language</span><span class="pill">Sentiment</span><span class="pill">Emotion</span><span class="pill">Sarcasm</span></div>
    </div>
    <div class="footer-note">Private by design · Analysis runs on the file you provide</div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # ── NLP Enrichment with caching ───────────────────────────────────────────
    # cached_enrich() runs sentiment + emotion inference on every message.
    # The result is cached by the MD5 hash of the uploaded file.
    # First upload  : runs both models, shows a progress bar, takes time.
    # Same file again: returns the cached result instantly — no inference.
    # Different file : detects the new hash and runs inference again.
    # A real progress bar (0% → 100%) replaces the vague spinner so the user can see exactly how far along the analysis is.
    file_hash = hashlib.md5(bytes_data).hexdigest()
    df = nlp_analyzer.cached_enrich(file_hash, df)
    
    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Focus analysis on",user_list)
    display_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    st.markdown(f"""
    <div class="hero">
        <div class="hero-kicker">Conversation loaded</div>
        <h1>{'The whole conversation' if selected_user == 'Overall' else selected_user}</h1>
        <p>{len(display_df):,} messages ready to explore. Move from the big picture to the small signals: who speaks, when the chat moves, and what the language feels like.</p>
        <div class="hero-meta"><span class="pill">{len(user_list) - 1} participants</span><span class="pill">3 AI lenses</span><span class="pill">{display_df['date'].min().strftime('%b %Y')} – {display_df['date'].max().strftime('%b %Y')}</span></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Preview parsed messages", expanded=False):
        st.dataframe(display_df, use_container_width=True, height=260)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.header("At a glance")
        st.markdown('<div class="section-intro">A quick read on the scale and shape of this conversation.</div>', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="glance-grid">
            <div class="glance-card"><div class="glance-label">Total messages</div><div class="glance-value">{num_messages:,}</div></div>
            <div class="glance-card"><div class="glance-label">Total words</div><div class="glance-value">{words:,}</div></div>
            <div class="glance-card"><div class="glance-label">Media shared</div><div class="glance-value">{num_media_messages:,}</div></div>
            <div class="glance-card"><div class="glance-label">Links shared</div><div class="glance-value">{num_links:,}</div></div>
        </div>
        <div class="timeline-heading"></div>
        ''', unsafe_allow_html=True)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=45)
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        if user_heatmap.empty:
            st.warning("No activity data to show for the heatmap.")
        else:
            fig,ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)
        fig,ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")
        col1,col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%1.0f%%",
                   wedgeprops={"linewidth": 2, "edgecolor": "white"},
                   textprops={"color": "#1d2733", "fontsize": 9})
            st.pyplot(fig)

        # Sentiment Analysis
        st.title("Sentiment Analysis")
        st.caption(
            "Sentiment analysis using a multilingual XLM-RoBERTa model. "
            "The score represents model confidence (0–1), not emotional intensity."
        )

        sentiment_df = df if selected_user == "Overall" else df[df["user"] == selected_user]
        sentiment_df = sentiment_df[sentiment_df["sentiment"].notna()]

        if sentiment_df.empty:
            st.warning("No analyzable messages found for sentiment analysis.")
        else:
            total_analyzed = len(sentiment_df)
            counts = sentiment_df["sentiment"].value_counts()

            pos_count = counts.get("Positive", 0)
            neu_count = counts.get("Neutral",  0)
            neg_count = counts.get("Negative", 0)
            pos_pct = round(pos_count / total_analyzed * 100, 1)
            neu_pct = round(neu_count / total_analyzed * 100, 1)
            neg_pct = round(neg_count / total_analyzed * 100, 1)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Messages Analyzed", total_analyzed)
            with col2:
                st.metric("Positive", f"{pos_pct}%", delta=f"{pos_count} msgs")
            with col3:
                st.metric("Neutral",  f"{neu_pct}%", delta=f"{neu_count} msgs")
            with col4:
                st.metric("Negative", f"{neg_pct}%", delta=f"{neg_count} msgs")

            col1, col2 = st.columns(2)

            with col1:
                st.header("Sentiment Distribution")
                sentiment_order  = ["Positive", "Neutral", "Negative"]
                sentiment_colors = {"Positive": "green", "Neutral": "grey", "Negative": "red"}
                bar_labels = [s for s in sentiment_order if s in counts.index]
                bar_values = [counts[s] for s in bar_labels]
                bar_colors = [sentiment_colors[s] for s in bar_labels]
                fig, ax = plt.subplots()
                ax.bar(bar_labels, bar_values, color=bar_colors)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Sentiment Proportion")
                fig, ax = plt.subplots()
                ax.pie(bar_values, labels=bar_labels,
                      colors=bar_colors, autopct="%1.0f%%",
                      wedgeprops={"linewidth": 2, "edgecolor": "white"},
                      textprops={"color": "#1d2733", "fontsize": 9})
                st.pyplot(fig)

            st.header("Average Model Confidence")
            st.caption("How certain the model was per label. 1.0 = fully certain, 0.5 = uncertain.")
            avg_conf = (
                sentiment_df.groupby("sentiment")["sentiment_score"]
                .mean().round(3).reset_index()
                .rename(columns={"sentiment": "Sentiment", "sentiment_score": "Avg Confidence"})
            )
            order_map = {"Positive": 0, "Neutral": 1, "Negative": 2}
            avg_conf["_order"] = avg_conf["Sentiment"].map(order_map)
            avg_conf = avg_conf.sort_values("_order").drop(columns="_order").reset_index(drop=True)
            st.dataframe(avg_conf, use_container_width=True)

            st.info(
                "Hinglish messages may contain spelling variations and code-switching. "
                "Short messages can be ambiguous. Sentiment confidence represents model certainty, "
                "not emotional intensity."
            )
        
        # Emotion Analysis
        st.title("Emotion Analysis")
        st.caption(
            "Multilingual XLM-RoBERTa emotion classification model. "
            "The model predicts 11 emotion categories. "
            "Emotion score represents model confidence (0–1), not emotional intensity."
        )

        EMOTION_COLORS = nlp_analyzer.EMOTION_COLORS
        EMOTION_ICONS  = nlp_analyzer.EMOTION_ICONS

        emotion_df = df if selected_user == "Overall" else df[df["user"] == selected_user]
        emotion_df = emotion_df[emotion_df["emotion"].notna()]

        if emotion_df.empty:
            st.warning("No analyzable messages found for emotion analysis.")
        else:
            total_emotion  = len(emotion_df)
            emotion_counts = emotion_df["emotion"].value_counts()
            detected_emotions = emotion_counts.index.tolist()

            st.header("Emotion Breakdown")
            cols = st.columns(min(4, len(detected_emotions)))
            for i, emo in enumerate(detected_emotions):
                col_idx = i % 4
                if i > 0 and col_idx == 0:
                    cols = st.columns(4)
                count = emotion_counts[emo]
                pct   = round(count / total_emotion * 100, 1)
                icon  = EMOTION_ICONS.get(emo, "")
                with cols[col_idx]:
                    st.metric(label=f"{icon} {emo}", value=f"{pct}%", delta=f"{count} msgs")

            col1, col2 = st.columns(2)

            with col1:
                st.header("Emotion Distribution")
                bar_labels = emotion_counts.index.tolist()
                bar_values = emotion_counts.values.tolist()
                bar_colors = [EMOTION_COLORS.get(e, "#aaaaaa") for e in bar_labels]
                fig, ax = plt.subplots()
                # Plot each bar individually so matplotlib cannot override colours
                for label, value, color in zip(bar_labels, bar_values, bar_colors):
                    ax.bar(label, value, color=color)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Emotion Proportion")
                fig, ax = plt.subplots()
                ax.pie(bar_values, labels=bar_labels,
                      colors=bar_colors, autopct="%1.0f%%",
                      wedgeprops={"linewidth": 2, "edgecolor": "white"},
                      textprops={"color": "#1d2733", "fontsize": 9})
                st.pyplot(fig)

            st.header("Sentiment x Emotion")
            st.caption("How many messages fall into each Sentiment + Emotion combination.")
            cross_df = df if selected_user == "Overall" else df[df["user"] == selected_user]
            cross_df = cross_df[cross_df["sentiment"].notna() & cross_df["emotion"].notna()]

            if cross_df.empty:
                st.info("Not enough data for the cross-analysis heatmap.")
            else:
                cross_table = pd.crosstab(cross_df["sentiment"], cross_df["emotion"])
                sent_order  = [s for s in ["Positive", "Neutral", "Negative"] if s in cross_table.index]
                cross_table = cross_table.loc[sent_order]
                fig, ax = plt.subplots()
                sns.heatmap(cross_table, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            if selected_user == "Overall":
                st.info("Per-user emotion breakdown is shown in the 'Emotion by User' section below.")

            st.header("Average Confidence per Emotion")
            avg_emo_conf = (
                emotion_df.groupby("emotion")["emotion_score"]
                .mean().round(3).reset_index()
                .rename(columns={"emotion": "Emotion", "emotion_score": "Avg Confidence"})
                .sort_values("Avg Confidence", ascending=False).reset_index(drop=True)
            )
            avg_emo_conf.insert(0, "Icon",
                avg_emo_conf["Emotion"].map(lambda e: EMOTION_ICONS.get(e, "")))
            st.dataframe(avg_emo_conf, use_container_width=True)

            st.info(
                "Emotion predictions are based on 11 predefined emotion categories. "
                "Short or ambiguous Hinglish messages may be classified as Neutral. "
                "Confidence indicates model certainty, not emotional intensity."
            )
        
        # Emoji + Emotion
        st.title("Emoji + Emotion Analysis")
        st.caption("Which emojis appear alongside each predicted emotion in this chat.")

        pair_df, pivot_df, top_per_emotion_df = helper.emoji_emotion_analysis(selected_user, df)

        if pair_df.empty:
            st.warning("No emoji + emotion data found. Messages may have no emojis.")
        else:
            EMOTION_COLORS = nlp_analyzer.EMOTION_COLORS
            EMOTION_ICONS  = nlp_analyzer.EMOTION_ICONS

            st.header("Top Emojis per Emotion")
            emotions_present = sorted(top_per_emotion_df["emotion"].unique().tolist())
            if emotions_present:
                num_cols = min(3, len(emotions_present))
                cols = st.columns(num_cols)
                for i, emo in enumerate(emotions_present):
                    col_idx = i % num_cols
                    if i > 0 and col_idx == 0:
                        cols = st.columns(num_cols)
                    emo_data = top_per_emotion_df[
                        top_per_emotion_df["emotion"] == emo
                    ].reset_index(drop=True)
                    icon  = EMOTION_ICONS.get(emo, "")
                    color = EMOTION_COLORS.get(emo, "#bdc3c7")
                    with cols[col_idx]:
                        st.markdown(
                            f"<div style='background:{color}22;border-left:4px solid {color};"
                            f"padding:8px;border-radius:4px;'><b>{icon} {emo}</b></div>",
                            unsafe_allow_html=True
                        )
                        for _, row in emo_data.iterrows():
                            st.markdown(
                                f"&nbsp;&nbsp;`{row['emoji']}`  x{int(row['count'])}",
                                unsafe_allow_html=True
                            )

            st.header("Emoji x Emotion Heatmap")
            st.caption("Top 15 emojis vs emotions. Darker = more co-occurrences.")
            if not pivot_df.empty:
                fig, ax = plt.subplots()
                sns.heatmap(pivot_df, annot=True, fmt="g", cmap="YlOrRd", ax=ax)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.header("Emoji x Emotion Table")
            display_pair = pair_df.head(50).copy()
            display_pair.insert(1, "icon",
                display_pair["emotion"].map(lambda e: EMOTION_ICONS.get(e, "")))
            display_pair = display_pair.rename(columns={
                "emoji": "Emoji", "icon": "Icon",
                "emotion": "Emotion", "count": "Count"
            })
            st.dataframe(display_pair, use_container_width=True)

            st.info(
                "Co-occurrence means the emoji appeared in a message predicted with that emotion. "
                "The same emoji can appear across multiple emotions depending on context."
            )
        
        # Sentiment by User + Emotion by User
        st.title("Sentiment by User")
        st.caption("% of each user's messages as Positive / Neutral / Negative. Min 5 messages to appear.")

        sent_colors  = {"Positive": "green", "Neutral": "grey", "Negative": "red"}
        sent_pivot, sent_counts = helper.sentiment_by_user(selected_user, df)

        if sent_pivot.empty:
            st.warning("Not enough data. Need at least 5 analyzable messages per user.")
        else:
            x     = range(len(sent_pivot))
            width = 0.25
            fig, ax = plt.subplots()
            for i, sentiment in enumerate(["Positive", "Neutral", "Negative"]):
                if sentiment not in sent_pivot.columns:
                    continue
                offsets = [xi + (i - 1) * width for xi in x]
                ax.bar(offsets, sent_pivot[sentiment], width=width,
                       label=sentiment, color=sent_colors[sentiment])
            ax.set_xticks(list(x))
            ax.set_xticklabels(sent_pivot.index, rotation='vertical')
            ax.set_ylabel("% of Messages")
            ax.legend(title="Sentiment")
            st.pyplot(fig)

            with st.expander("Show raw message counts"):
                counts_display = sent_counts.copy()
                counts_display["Total"] = counts_display.sum(axis=1)
                st.dataframe(counts_display, use_container_width=True)

        st.title("Emotion by User")
        st.caption("% of each user's messages per emotion. Min 5 messages to appear.")

        EMOTION_COLORS = nlp_analyzer.EMOTION_COLORS
        EMOTION_ICONS  = nlp_analyzer.EMOTION_ICONS
        emo_pivot, emo_counts = helper.emotion_by_user(selected_user, df)

        if emo_pivot.empty:
            st.warning("Not enough data. Need at least 5 analyzable messages per user.")
        else:
            emotion_col_order = [
                e for e in ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"]
                if e in emo_pivot.columns
            ]
            for col in emo_pivot.columns:
                if col not in emotion_col_order:
                    emotion_col_order.append(col)

            emo_pivot_ordered = emo_pivot[emotion_col_order]
            bar_colors_emo    = [EMOTION_COLORS.get(e, "grey") for e in emotion_col_order]

            fig, ax = plt.subplots()
            # Use matplotlib directly instead of pivot.plot() —
            # pandas .plot() ignores the color list in some versions.
            lefts = [0.0] * len(emo_pivot_ordered)
            for emo, color in zip(emotion_col_order, bar_colors_emo):
                values = emo_pivot_ordered[emo].values
                ax.barh(emo_pivot_ordered.index, values, left=lefts,
                        color=color, label=emo)
                lefts = [l + v for l, v in zip(lefts, values)]
            ax.set_xlabel("% of Messages")
            ax.legend(title="Emotion", bbox_to_anchor=(1.01, 1), loc="upper left")
            st.pyplot(fig)

            st.header("Most Dominant Emotion per User")
            dominant = emo_pivot_ordered.idxmax(axis=1).reset_index()
            dominant.columns = ['User', 'Dominant Emotion']
            dominant['% of Messages'] = dominant.apply(
                lambda row: f"{emo_pivot_ordered.loc[row['User'], row['Dominant Emotion']]:.1f}%",
                axis=1
            )
            dominant.insert(2, "Icon",
                dominant["Dominant Emotion"].map(lambda e: EMOTION_ICONS.get(e, "")))
            st.dataframe(dominant, use_container_width=True)

            with st.expander("Show raw message counts"):
                counts_emo_display = emo_counts[emotion_col_order].copy()
                counts_emo_display["Total"] = counts_emo_display.sum(axis=1)
                st.dataframe(counts_emo_display, use_container_width=True)

            st.info(
                "Percentages based only on analyzable messages. "
                "Users with fewer than 5 messages are hidden. "
                "Dominant emotion is the most frequent prediction, not a personality assessment."
            )
            
        # Temporal Analysis
        EMOTION_COLORS = nlp_analyzer.EMOTION_COLORS
        EMOTION_ICONS  = nlp_analyzer.EMOTION_ICONS

        st.title("Sentiment Over Time")
        st.caption("Monthly % of Positive / Neutral / Negative messages.")

        sot_df = helper.sentiment_over_time(selected_user, df)

        if sot_df.empty:
            st.warning("Not enough sentiment data to show a timeline.")
        else:
            fig, ax = plt.subplots()
            ax.plot(sot_df["time_label"], sot_df["Positive"], color='green',  marker='o', label='Positive')
            ax.plot(sot_df["time_label"], sot_df["Neutral"],  color='grey',   marker='o', label='Neutral')
            ax.plot(sot_df["time_label"], sot_df["Negative"], color='red',    marker='o', label='Negative')
            plt.xticks(rotation='vertical')
            ax.set_ylabel("% of Messages")
            ax.legend(title="Sentiment")
            st.pyplot(fig)

            with st.expander("Show monthly sentiment table"):
                st.dataframe(
                    sot_df.rename(columns={"time_label": "Month", "total": "Messages Analyzed"}),
                    use_container_width=True
                )

        st.title("Emotion Over Time")
        st.caption("Monthly % of the top 5 most common emotions (as % of all analyzed messages that month).")

        eot_df, top_emotions = helper.emotion_over_time(selected_user, df, top_n=5)

        if eot_df.empty:
            st.warning("Not enough emotion data to show a timeline.")
        else:
            emo_cols_present = [e for e in top_emotions if e in eot_df.columns]
            # Build the explicit colour list — one colour per emotion layer.
            # We build this BEFORE the plot call so we can print/debug it if needed.
            stack_colors = [EMOTION_COLORS.get(e, "#aaaaaa") for e in emo_cols_present]
            fig, ax = plt.subplots()
            ax.stackplot(
                eot_df["time_label"],
                [eot_df[e].values for e in emo_cols_present],
                labels=[f"{EMOTION_ICONS.get(e,'')} {e}" for e in emo_cols_present],
                colors=stack_colors
            )
            plt.xticks(rotation='vertical')
            ax.set_ylabel("% of Messages")
            ax.legend(title="Emotion", bbox_to_anchor=(1.01, 1), loc="upper left")
            st.pyplot(fig)

            with st.expander("Show monthly emotion table"):
                st.dataframe(
                    eot_df.rename(columns={"time_label": "Month"}),
                    use_container_width=True
                )

        st.title("Sentiment Trend Score")
        st.caption(
            "Monthly average: Positive=+1, Neutral=0, Negative=-1. "
            "Score near +1 = mostly positive month, near -1 = mostly negative."
        )
        st.info(
            "Statistical representation of predicted sentiment polarity — "
            "not a measure of actual mood or psychological state."
        )

        trend_df = helper.sentiment_trend(selected_user, df)

        if trend_df.empty:
            st.warning("Not enough data to calculate sentiment trend.")
        else:
            bar_colors_tr = [
                'green' if v >= 0.1 else 'red' if v <= -0.1 else 'grey'
                for v in trend_df["trend_score"]
            ]
            fig, ax = plt.subplots()
            ax.bar(trend_df["time_label"], trend_df["trend_score"], color=bar_colors_tr)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xticks(rotation='vertical')
            ax.set_ylabel("Avg Sentiment Score (-1 to +1)")
            st.pyplot(fig)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Month",
                    trend_df.loc[trend_df["trend_score"].idxmax(), "time_label"],
                    delta=f"{trend_df['trend_score'].max():+.2f}")
            with col2:
                st.metric("Worst Month",
                    trend_df.loc[trend_df["trend_score"].idxmin(), "time_label"],
                    delta=f"{trend_df['trend_score'].min():+.2f}")
            with col3:
                st.metric("Overall Average", f"{trend_df['trend_score'].mean():+.2f}")
            with col4:
                st.metric("Months Analyzed", len(trend_df))

            with st.expander("Show monthly trend table"):
                st.dataframe(
                    trend_df.rename(columns={
                        "time_label": "Month",
                        "trend_score": "Trend Score (-1 to +1)",
                        "total": "Messages Analyzed"
                    }),
                    use_container_width=True
                )
        
        # Top Positive and Negative Messages
        # Shows the individual messages the model was most confident about.
        # Useful for sanity-checking the model and exploring the data.
        # sentiment_score = model confidence (0–1), NOT emotional intensity.
        # ======================================================================
        st.title("Most Positive and Negative Messages")
        st.caption(
            "Messages the sentiment model was most confident about. "
            "Confidence score = model certainty (0–1), not emotional intensity. "
            "A score of 0.99 means the model was 99% sure about that label."
        )

        top_pos, top_neg = helper.top_positive_negative_messages(selected_user, df, n=10)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Top 10 Positive Messages")
            if top_pos.empty:
                st.info("No positive messages found.")
            else:
                st.dataframe(
                    top_pos.rename(columns={
                        "user":            "User",
                        "date":            "Date",
                        "message":         "Message",
                        "sentiment":       "Sentiment",
                        "sentiment_score": "Confidence"
                    }),
                    use_container_width=True
                )

        with col2:
            st.header("Top 10 Negative Messages")
            if top_neg.empty:
                st.info("No negative messages found.")
            else:
                st.dataframe(
                    top_neg.rename(columns={
                        "user":            "User",
                        "date":            "Date",
                        "message":         "Message",
                        "sentiment":       "Sentiment",
                        "sentiment_score": "Confidence"
                    }),
                    use_container_width=True
                )

        st.info(
            "**Note:** These tables contain real message text from your chat. "
            "High confidence does not mean the message is more important — "
            "it only means the model had less doubt about its prediction. "
            "Short, unambiguous messages (e.g. 'I hate this') often score highest."
        )
        
        # Sarcasm / irony detection.
        # The model performs binary classification:
        # LABEL_0 → Not Sarcastic
        # LABEL_1 → Sarcastic
        # sarcasm_confidence represents confidence in the predicted label.
        st.title("Sarcasm / Irony Detection")
        st.caption(
            "Sarcasm detection using an XLM-RoBERTa-based classification model. "
            "The sarcasm score represents model confidence (0–1) for the predicted label."
        )
        st.warning(
            "**Note:** Sarcasm is highly context-dependent and may require conversational context. "
            "The model analyzes messages individually, so subtle or context-dependent sarcasm "
            "may be missed. Predictions should be treated as statistical estimates."
        )

        sarc_df = df if selected_user == "Overall" else df[df["user"] == selected_user]
        sarc_df = sarc_df[sarc_df["is_sarcastic"].notna()]

        if sarc_df.empty:
            st.warning("No sarcasm analysis data found. Re-upload the file to run the model.")
        else:
            total_sarc   = len(sarc_df)
            sarc_count   = int(sarc_df["is_sarcastic"].sum())
            non_count    = total_sarc - sarc_count
            sarc_pct     = round(sarc_count / total_sarc * 100, 1)
            non_pct      = round(non_count  / total_sarc * 100, 1)

            # Metric cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Messages Analyzed", total_sarc)
            with col2:
                st.metric("Sarcastic", f"{sarc_pct}%", delta=f"{sarc_count} msgs")
            with col3:
                st.metric("Not Sarcastic", f"{non_pct}%", delta=f"{non_count} msgs")

            # Distribution bar + pie
            col1, col2 = st.columns(2)

            with col1:
                st.header("Sarcasm Distribution")
                sarc_dist = helper.sarcasm_distribution(selected_user, df)
                fig, ax   = plt.subplots()
                colors    = ['#e74c3c' if l == 'Sarcastic' else '#95a5a6'
                             for l in sarc_dist['label']]
                ax.bar(sarc_dist['label'], sarc_dist['count'], color=colors)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Sarcasm Proportion")
                fig, ax = plt.subplots()
                ax.pie(
                    sarc_dist['count'],
                    labels=sarc_dist['label'],
                    colors=colors,
                    autopct="%1.0f%%",
                    wedgeprops={"linewidth": 2, "edgecolor": "white"},
                    textprops={"color": "#1d2733", "fontsize": 9}
                )
                st.pyplot(fig)

            # Sarcasm rate by user (Overall only)
            if selected_user == "Overall":
                st.header("Sarcasm Rate by User")
                st.caption("% of each user's messages predicted as sarcastic. Min 5 messages to appear.")

                sarc_user_df = helper.sarcasm_by_user(selected_user, df)

                if sarc_user_df.empty:
                    st.info("Not enough data per user (need ≥ 5 analyzed messages).")
                else:
                    fig, ax = plt.subplots()
                    # Give each user a distinct colour from the tab10 palette
                    import matplotlib.cm as cm
                    n_users    = len(sarc_user_df)
                    tab_colors = [cm.tab10(i / max(n_users, 1)) for i in range(n_users)]
                    ax.bar(sarc_user_df['user'], sarc_user_df['sarcasm_rate'], color=tab_colors)
                    plt.xticks(rotation='vertical')
                    ax.set_ylabel("Sarcasm Rate (%)")
                    st.pyplot(fig)

                    with st.expander("Show raw counts per user"):
                        st.dataframe(
                            sarc_user_df.rename(columns={
                                "user":             "User",
                                "sarcastic_count":  "Sarcastic",
                                "total_analyzed":   "Total Analyzed",
                                "sarcasm_rate":     "Sarcasm Rate (%)"
                            }),
                            use_container_width=True
                        )

            # Sarcasm over time
            st.header("Sarcasm Rate Over Time")
            st.caption("Monthly % of messages predicted as sarcastic.")

            sarc_time_df = helper.sarcasm_over_time(selected_user, df)

            if sarc_time_df.empty:
                st.info("Not enough data for a timeline.")
            else:
                fig, ax = plt.subplots()
                ax.plot(sarc_time_df["time_label"], sarc_time_df["sarcasm_rate"],
                        color='#e74c3c', marker='o', label='Sarcasm Rate %')
                plt.xticks(rotation='vertical')
                ax.set_ylabel("Sarcasm Rate (%)")
                ax.legend()
                st.pyplot(fig)

                with st.expander("Show monthly sarcasm table"):
                    st.dataframe(
                        sarc_time_df.rename(columns={
                            "time_label":   "Month",
                            "sarcasm_rate": "Sarcasm Rate (%)",
                            "total":        "Messages Analyzed"
                        }),
                        use_container_width=True
                    )

            # Top sarcastic messages
            st.header("Most Confidently Sarcastic Messages")
            st.caption(
                "Messages the model was most confident are sarcastic. "
                "Includes their predicted sentiment and emotion for context."
            )

            top_sarc_msgs = helper.top_sarcastic_messages(selected_user, df, n=10)

            if top_sarc_msgs.empty:
                st.info("No sarcastic messages detected.")
            else:
                rename_map = {
                    "user":          "User",
                    "date":          "Date",
                    "message":       "Message",
                    "sarcasm_confidence": "Sarcasm Confidence",
                    "sentiment":     "Sentiment",
                    "emotion":       "Emotion"
                }
                st.dataframe(
                    top_sarc_msgs.rename(columns={k: v for k, v in rename_map.items()
                                                  if k in top_sarc_msgs.columns}),
                    use_container_width=True
                )

            # Sarcasm × Sentiment cross-analysis
            st.header("Sarcasm × Sentiment")
            st.caption(
                "Distribution of predicted sarcasm labels across sentiment categories. "
                "Sarcastic messages may receive positive sentiment predictions when "
                "the surface wording appears positive."
            )

            cross_sarc = df if selected_user == "Overall" else df[df["user"] == selected_user]
            cross_sarc = cross_sarc[
                cross_sarc["is_sarcastic"].notna() &
                cross_sarc["sentiment"].notna()
            ].copy()
            cross_sarc["Sarcasm"] = cross_sarc["is_sarcastic"].map(
                {True: "Sarcastic", False: "Not Sarcastic"}
            )

            if cross_sarc.empty:
                st.info("Not enough data.")
            else:
                cross_tbl = pd.crosstab(cross_sarc["Sarcasm"], cross_sarc["sentiment"])
                sent_order = [s for s in ["Positive", "Neutral", "Negative"]
                              if s in cross_tbl.columns]
                cross_tbl  = cross_tbl[sent_order]
                fig, ax    = plt.subplots()
                sns.heatmap(cross_tbl, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Disclaimer
            st.info(
                "**Sarcasm detection limitations:**  \n"
                "- Sarcasm is highly dependent on context and conversational intent.  \n"
                "- The model analyzes each message independently.  \n"
                "- Scores near 0.5 indicate greater model uncertainty.  \n"
                "- Sarcastic messages may receive positive sentiment predictions because "
                "sentiment and sarcasm are separate classification tasks.  \n"
                "- Predictions are statistical estimates, not human judgements."
            )
        