import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import random
from datetime import datetime, timedelta
from collections import Counter
import gradio as gr
from wordcloud import WordCloud
import os

from transformers import pipeline
from keybert import KeyBERT

from transformers import pipeline
from keybert import KeyBERT
from youtube_comment_downloader import YoutubeCommentDownloader
from datetime import datetime, timedelta
import re
import pandas as pd



# Initialize models globally
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
kw_model = KeyBERT()

# Label mapping - handling different model outputs
sentiment_map = {
    "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive",  # RoBERTa format
    "negative": "Negative", "neutral": "Neutral", "positive": "Positive",  # Standard format
    "NEGATIVE": "Negative", "NEUTRAL": "Neutral", "POSITIVE": "Positive"   # Uppercase format
}
color_map = {"Positive": "#2E8B57", "Neutral": "#4682B4", "Negative": "#CD5C5C"}



# Default comments for when no file is uploaded
comments = [
    "This new distance fare is really fair. I pay less for short trips!",
    "It's confusing, I don't know how much I'll pay now.",
    "RURA should have informed us better about this change.",
    "Good step towards fairness and modernization.",
    "Too expensive now! I hate this new system.",
    "The distance-based system makes so much more sense than flat rates.",
    "Why should I pay the same for 1km as I would for 10km? This is better.",
    "Finally a fair system — short-distance commuters benefit the most!",
    "I'm still unsure how the new rates are calculated. Needs clarity.",
    "A detailed public awareness campaign would have helped a lot.",
    "Smart move toward a fairer system, but more awareness is needed.",
    "I'm paying more now and it feels unjust.",
    "Flat rates were easier to understand, but this is more logical.",
    "Paying based on distance is reasonable, but it needs fine-tuning.",
    "App crashes when I try to calculate my fare. Fix it!",
    "Drivers are confused about the new system too.",
    "Great initiative but poor implementation.",
    "Now I know exactly what I'm paying for. Transparent and fair.",
    "The fare calculator is very helpful.",
    "Bus company profits will increase, but what about us passengers?",
    "I've noticed faster service since the new system launched.",
    "Rural areas are being charged too much now.",
    "The new system is making my daily commute more expensive.",
    "Distance-based fares are the future of transportation.",
    "I appreciate the transparency but the app needs work.",
    "This discriminates against people living in rural areas!",
    "My transportation costs have decreased by 30%!",
    "We should go back to the old system immediately.",
    "Kids going to school are now paying more, this is unfair.",
    "The government did a good job explaining the benefits.",
    "I've waited years for a fair pricing system like this.",
    "Very impressed with the new fare calculation technology.",
    "The app is too complicated for elderly passengers.",
    "The transition period should have been longer.",
    "I find the new fare calculator very intuitive.",
    "This is just another way to extract more money from us.",
    "Love how I can now predict exactly what my trip will cost.",
    "The implementation was rushed without proper testing.",
    "Prices vary too much depending on traffic congestion.",
    "Works well in urban areas but rural commuters are suffering.",
    "I've downloaded the fare calculator app - it's brilliant!",
    "Taxi drivers are confused about calculating fares correctly."
]

# Global variable to hold the current dataframe
global_df = None

# Function to generate default dataset from predefined comments

def generate_default_df():
    global global_df
    default_data = []
    start_time = datetime.now() - timedelta(hours=24)

    for i, comment in enumerate(comments):
        timestamp = start_time + timedelta(hours=random.uniform(0, 24))

        # Analyze sentiment
        result = classifier(comment)[0]
        sentiment = sentiment_map[result["label"]]
        score = round(result["score"], 3)

        # Extract keywords
        try:
            keywords = kw_model.extract_keywords(comment, top_n=3)
            keyword_str = ", ".join([kw[0] for kw in keywords]) if keywords else "N/A"
        except:
            keyword_str = "N/A"

        default_data.append({
            "Datetime": timestamp,
            "Text": comment,
            "Sentiment": sentiment,
            "Score": score,
            "Keywords": keyword_str
        })

    default_df = pd.DataFrame(default_data)
    default_df["Datetime"] = pd.to_datetime(default_df["Datetime"])
    default_df["Datetime"] = default_df["Datetime"].dt.floor("1H")
    global_df = default_df.sort_values("Datetime").reset_index(drop=True)
    return global_df



import re
from datetime import datetime, timedelta

def convert_relative_time(relative):
    now = datetime.now()
    try:
        match = re.match(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', relative.lower())
        if not match:
            return now  # fallback to now for unknown formats

        value, unit = int(match.group(1)), match.group(2)

        if unit == 'second':
            dt = now - timedelta(seconds=value)
        elif unit == 'minute':
            dt = now - timedelta(minutes=value)
        elif unit == 'hour':
            dt = now - timedelta(hours=value)
        elif unit == 'day':
            dt = now - timedelta(days=value)
        elif unit == 'week':
            dt = now - timedelta(weeks=value)
        elif unit == 'month':
            dt = now - timedelta(days=value * 30)
        elif unit == 'year':
            dt = now - timedelta(days=value * 365)
        else:
            dt = now
    except Exception as e:
        print(f"Failed to parse relative time '{relative}': {e}")
        dt = now
    return dt

def generate_df(comments):
    global global_df
    default_data = []

    for comment in comments:
        text = comment.get('text', '')
        timestamp = convert_relative_time(comment.get('time', '0 seconds ago'))

        # Truncate long text for model input (e.g. 512 tokens)
        truncated_text = text[:512]

        # Sentiment analysis
        try:
            result = classifier(truncated_text)[0]
            sentiment = sentiment_map.get(result["label"], "Unknown")
            score = round(result["score"], 3)
        except Exception as e:
            print(f"Sentiment classification failed: {e}")
            sentiment = "Unknown"
            score = 0.0

        # Keyword extraction
        try:
            keywords = kw_model.extract_keywords(truncated_text, top_n=3)
            keyword_str = ", ".join([kw[0] for kw in keywords]) if keywords else "N/A"
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            keyword_str = "N/A"

        default_data.append({
            "Datetime": timestamp,
            "Text": text,
            "Sentiment": sentiment,
            "Score": score,
            "Keywords": keyword_str
        })

    default_df = pd.DataFrame(default_data)
    default_df["Datetime"] = pd.to_datetime(default_df["Datetime"])
    default_df["Datetime"] = default_df["Datetime"].dt.floor("1H")
    global_df = default_df.sort_values("Datetime").reset_index(drop=True)
    return global_df





# Function to process uploaded CSV or Excel file and analyze sentiment

def process_uploaded_file(file):
    global global_df

    if file is None:
        global_df = generate_default_df()
        return global_df

    try:
        # Read the uploaded file
        if file.name.endswith('.csv'):
            user_df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            user_df = pd.read_excel(file.name)
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel files only.")

        # Check required columns
        if 'Text' not in user_df.columns:
            raise ValueError("File must contain a 'Text' column with comments.")

        # Handle datetime - create if not exists
        if 'Datetime' not in user_df.columns:
            # Generate timestamps for uploaded data
            start_time = datetime.now() - timedelta(hours=len(user_df))
            user_df['Datetime'] = [start_time + timedelta(hours=i) for i in range(len(user_df))]

        # Clean and prepare data
        user_df = user_df[['Datetime', 'Text']].copy()
        user_df["Datetime"] = pd.to_datetime(user_df["Datetime"])
        user_df["Datetime"] = user_df["Datetime"].dt.floor("1H")
        user_df = user_df.dropna(subset=['Text'])

        # Analyze sentiment and extract keywords for each comment
        sentiments = []
        scores = []
        keywords_list = []

        for text in user_df["Text"]:
            try:
                # Sentiment analysis
                result = classifier(str(text))[0]
                sentiment = sentiment_map[result['label']]
                score = round(result['score'], 3)

                # Keyword extraction
                keywords = kw_model.extract_keywords(str(text), top_n=3)
                keyword_str = ", ".join([kw[0] for kw in keywords]) if keywords else "N/A"

                sentiments.append(sentiment)
                scores.append(score)
                keywords_list.append(keyword_str)
            except Exception as e:
                print(f"Error processing text: {e}")
                sentiments.append("Neutral")
                scores.append(0.5)
                keywords_list.append("N/A")

        user_df["Sentiment"] = sentiments
        user_df["Score"] = scores
        user_df["Keywords"] = keywords_list

        global_df = user_df.sort_values("Datetime").reset_index(drop=True)
        return global_df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        global_df = generate_default_df()
        return global_df

# Function to wrapper function for file analysis to update dataframe display

def get_analysis_dataframe(file):
    return process_uploaded_file(file)

# Function to analyze a single comment and return sentiment and keywords

def analyze_text(comment):
    if not comment or not comment.strip():
        return "N/A", 0, "N/A"

    try:
        result = classifier(comment)[0]
        sentiment = sentiment_map.get(result["label"], result["label"])
        score = result["score"]

        keywords = kw_model.extract_keywords(comment, top_n=3, keyphrase_ngram_range=(1, 2))
        keywords_str = ", ".join([kw[0] for kw in keywords]) if keywords else "N/A"

        return sentiment, score, keywords_str
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return "Error", 0, "Error processing text"

# Function to add analyzed comment to global dataframe

def add_to_dataframe(comment, sentiment, score, keywords):

    global global_df
    timestamp = datetime.now().replace(microsecond=0)

    new_row = pd.DataFrame([{
        "Datetime": timestamp,
        "Text": comment,
        "Sentiment": sentiment,
        "Score": score,
        "Keywords": keywords
    }])

    global_df = pd.concat([global_df, new_row], ignore_index=True)
    return global_df

# Function to generate and display a simple word cloud based on sentiment filter

def create_wordcloud_simple(df, sentiment_filter=None):
    if df is None or df.empty:
        return None

    # Filter by sentiment if provided
    if sentiment_filter and sentiment_filter != "All":
        filtered_df = df[df["Sentiment"] == sentiment_filter]
    else:
        filtered_df = df

    if filtered_df.empty:
        print("No data available for the selected sentiment.")
        return None

    # Combine keywords into a single string
    keyword_text = filtered_df["Keywords"].fillna("").str.replace("N/A", "").str.replace(",", " ")
    all_keywords = " ".join(keyword_text)

    if not all_keywords.strip():
        print("No valid keywords to display in word cloud.")
        return None

    # Select colormap based on sentiment
    colormap = "viridis"
    if sentiment_filter == "Positive":
        colormap = "Greens"
    elif sentiment_filter == "Neutral":
        colormap = "Blues"
    elif sentiment_filter == "Negative":
        colormap = "Reds"

    # Create the word cloud
    wordcloud = WordCloud(
        background_color='white',
        colormap=colormap,
        max_words=50,
        height=500,
    ).generate(all_keywords)

    # Convert to image for Gradio
    return wordcloud.to_image()



# Function to create a scatter plot showing comment volume by sentiment over time
def plot_sentiment_timeline(df):
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available", height=400)

    try:
        df_copy = df.copy()
        df_copy["Datetime"] = pd.to_datetime(df_copy["Datetime"])
        df_copy["Time_Bin"] = df_copy["Datetime"].dt.floor("1H")

        # Grouping comments by time and sentiment
        grouped = (
            df_copy.groupby(["Time_Bin", "Sentiment"])
            .agg(
                Count=("Text", "count"),
                Score=("Score", "mean"),
                Keywords=("Keywords", lambda x: ", ".join(set(", ".join(x).split(", "))) if len(x) > 0 else "")
            )
            .reset_index()
        )

        fig = go.Figure()

        for sentiment, color in color_map.items():
            sentiment_df = grouped[grouped["Sentiment"] == sentiment]
            if sentiment_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=sentiment_df["Time_Bin"],
                    y=sentiment_df["Count"],
                    mode='markers',
                    name=sentiment,
                    marker=dict(size=10, color=color, opacity=0.9, line=dict(width=1, color='DarkSlateGrey')),
                    text=sentiment_df["Keywords"],
                    hovertemplate='<b>%{y} comments</b><br>%{x}<br><b>Keywords:</b> %{text}<extra></extra>'
                )
            )

        fig.update_layout(
            title="Sentiment Distribution Over Time (1-Hour Bins)",
            height=500,
            xaxis=dict(
                title="Time",
                type="date",
                rangeslider=dict(visible=False),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            ),
            yaxis=dict(title="Number of Comments"),
            template="plotly_white"
        )

        return fig

    except Exception as e:
        print(f"Error in timeline plot: {e}")
        return go.Figure().update_layout(
            title="Error creating timeline visualization",
            height=400
        )



# Function to create a dual-view visualization of sentiment distribution

def plot_sentiment_distribution(df):
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available", height=400)

    try:
        # Group sentiment counts
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        sentiment_counts["Percentage"] = sentiment_counts["Count"] / sentiment_counts["Count"].sum() * 100

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Sentiment Distribution", "Sentiment Counts"),
            column_widths=[0.5, 0.5]
        )

        # Pie Chart
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts["Sentiment"],
                values=sentiment_counts["Count"],
                textinfo="percent+label",
                marker=dict(colors=[color_map.get(s, "#999999") for s in sentiment_counts["Sentiment"]]),
                hole=0.4
            ),
            row=1, col=1
        )

        # Bar Chart
        fig.add_trace(
            go.Bar(
                x=sentiment_counts["Sentiment"],
                y=sentiment_counts["Count"],
                text=sentiment_counts["Count"],
                textposition="auto",
                marker_color=[color_map.get(s, "#999999") for s in sentiment_counts["Sentiment"]]
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title="Sentiment Distribution Overview",
            height=500,
            autosize=True,
            width=None ,
            template="plotly_white",
            showlegend=False
        )

        return fig

    except Exception as e:
        print(f"Error in distribution plot: {e}")
        return go.Figure().update_layout(
            title="Error creating distribution visualization",
            height=500,
            autosize=True,
            width=None
        )

# Function to create a grouped bar chart visualization of the top keywords across sentiments

def plot_keyword_analysis(df):
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available", height=400)

    try:
        all_keywords = []

        # Process each sentiment
        for sentiment in ["Positive", "Neutral", "Negative"]:
            sentiment_df = df[df["Sentiment"] == sentiment]
            if sentiment_df.empty:
                continue

            # Extract and flatten keyword lists
            for keywords_str in sentiment_df["Keywords"].dropna():
                if keywords_str and keywords_str.upper() != "N/A":
                    keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                    for kw in keywords:
                        all_keywords.append((kw, sentiment))

        if not all_keywords:
            return go.Figure().update_layout(
                title="No keyword data available",
                height=500,
                autosize=True,
                width=None
            )

        # Create DataFrame and aggregate keyword counts
        keywords_df = pd.DataFrame(all_keywords, columns=["Keyword", "Sentiment"])
        keyword_counts = (
            keywords_df.groupby(["Keyword", "Sentiment"])
            .size()
            .reset_index(name="Count")
        )

        # Filter top 15 keywords by overall frequency
        top_keywords = keywords_df["Keyword"].value_counts().nlargest(15).index
        keyword_counts = keyword_counts[keyword_counts["Keyword"].isin(top_keywords)]

        # Plot grouped bar chart
        fig = px.bar(
            keyword_counts,
            x="Keyword",
            y="Count",
            color="Sentiment",
            color_discrete_map=color_map,
            text="Count",
            barmode="group",
            labels={"Count": "Frequency", "Keyword": ""},
            title="🔍 Top Keywords by Sentiment"
        )

        fig.update_layout(
            legend_title="Sentiment",
            xaxis=dict(categoryorder="total descending"),
            yaxis=dict(title="Frequency"),
            height=500,
            autosize=True,
            width=None ,
            template="plotly_white"
        )

        return fig

    except Exception as e:
        print(f"Error in keyword analysis: {e}")
        return go.Figure().update_layout(
            title="Error creating keyword visualization",
            height=500,
            autosize=True,
            width=None
        )

# Function to generate summary sentiment metrics for dashboard visualization

def create_summary_metrics(df):
    if df is None or df.empty:
        return {
            "total": 0, "positive": 0, "neutral": 0, "negative": 0,
            "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0,
            "sentiment_ratio": 0.0, "trend": "No data"
        }

    try:
        total_comments = len(df)

        # Count sentiments
        sentiment_counts = df["Sentiment"].value_counts().to_dict()
        positive = sentiment_counts.get("Positive", 0)
        neutral = sentiment_counts.get("Neutral", 0)
        negative = sentiment_counts.get("Negative", 0)

        # Calculate percentages safely
        def pct(count):
            return round((count / total_comments) * 100, 1) if total_comments else 0.0

        positive_pct = pct(positive)
        neutral_pct = pct(neutral)
        negative_pct = pct(negative)

        # Sentiment ratio (Positive : Negative)
        sentiment_ratio = round(positive / negative, 2) if negative > 0 else float('inf')

        # Trend detection based on time-series sentiment evolution
        trend = "Insufficient data"
        if total_comments >= 5 and "Datetime" in df.columns:
            sorted_df = df.sort_values("Datetime")
            mid = total_comments // 2
            first_half = sorted_df.iloc[:mid]
            second_half = sorted_df.iloc[mid:]

            # Compute positive sentiment proportion in both halves
            first_pos_pct = (first_half["Sentiment"] == "Positive").mean()
            second_pos_pct = (second_half["Sentiment"] == "Positive").mean()

            delta = second_pos_pct - first_pos_pct
            if delta > 0.05:
                trend = "Improving"
            elif delta < -0.05:
                trend = "Declining"
            else:
                trend = "Stable"

        return {
            "total": total_comments,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "positive_pct": positive_pct,
            "neutral_pct": neutral_pct,
            "negative_pct": negative_pct,
            "sentiment_ratio": sentiment_ratio,
            "trend": trend,
        }

    except Exception as e:
        print(f"Error in summary metrics: {e}")
        return {
            "total": 0, "positive": 0, "neutral": 0, "negative": 0,
            "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0,
            "sentiment_ratio": 0.0, "trend": "Error calculating"
        }

# Function to analyze a single comment for the Quick Analyzer tab

def gradio_analyze_comment(comment):

    try:
        if not comment or not comment.strip():
            return "N/A", "0.0%", "N/A"

        sentiment, score, keywords = analyze_text(comment)
        score_str = f"{score * 100:.1f}%"

        return sentiment, score_str, keywords

    except Exception as e:
        print(f"Error in gradio_analyze_comment: {e}")
        return "Error", "0.0%", "Error processing comment"

# Function to add a comment to the dashboard

def gradio_add_comment(comment):
    global global_df

    if not comment or not comment.strip():
        return global_df, "Please enter a comment", "", plot_sentiment_timeline(global_df), plot_sentiment_distribution(global_df), plot_keyword_analysis(global_df)

    sentiment, score, keywords = analyze_text(comment)
    updated_df = add_to_dataframe(comment, sentiment, score, keywords)

    # Generate feedback message
    feedback = f"✓ Added: {sentiment} comment (Confidence: {score*100:.1f}%)"


    # Update all visualizations
    timeline_plot = plot_sentiment_timeline(updated_df)
    distribution_plot = plot_sentiment_distribution(updated_df)
    keyword_plot = plot_keyword_analysis(updated_df)

    return updated_df, feedback, "", timeline_plot, distribution_plot, keyword_plot

# Function  to generate a word cloud image from the DataFrame

def gradio_generate_wordcloud(sentiment_filter):
    try:
        filter_value = sentiment_filter if sentiment_filter != "All" else None
        return create_wordcloud_simple(global_df, filter_value)
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None



# Function to export the current dataframe to CSV for download

def export_data_to_csv(df_component):
    global global_df
    try:
        if global_df is not None and not global_df.empty:
            csv_buffer = io.StringIO()
            global_df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            # Save to a temporary file
            filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            return filename
        else:
            return None
    except Exception as e:
        print(f"Error exporting data: {e}")
        return None



def analyze_youtube_comments(video_url):
    from youtube_comment_downloader import YoutubeCommentDownloader
    import re

    # Simple YouTube video URL validation
    youtube_pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}"
    if not re.match(youtube_pattern, video_url):
        raise gr.Error("🚫 Please provide a valid YouTube video link.")

    try:
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video_url)
        if not comments:
            raise gr.Error("⚠️ No comments found for this video.")
        return generate_df(comments)
    except Exception as e:
        raise gr.Error(f"❌ Failed to retrieve comments: {str(e)}")

# Global default data initialization
global_df = generate_default_df()


# Function: Load either a file or a video URL, return dashboard-ready components
def load_and_update_all_components(file, video_url):
    global global_df

    # Determine which input to process
    if file is not None:
        updated_df = get_analysis_dataframe(file)
    elif video_url:
        updated_df = analyze_youtube_comments(video_url)
    else:
        updated_df = global_df  # fallback to default data if nothing provided

    # Generate updated metrics and visuals
    metrics = create_summary_metrics(updated_df)
    global_df = updated_df  # Update global state

    return (
        updated_df,
        metrics["total"], metrics["positive_pct"], metrics["neutral_pct"],
        metrics["negative_pct"], metrics["sentiment_ratio"], metrics["trend"],
        plot_sentiment_timeline(updated_df),
        plot_sentiment_distribution(updated_df),
        plot_keyword_analysis(updated_df),
        updated_df
    )


import signal
import sys

def signal_handler(sig, frame):
    print('Gracefully shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)



# Create the Gradio interface and dashboard
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # A Smart Dashboard for Analyzing Public Sentiment and Perception
        #### This interactive dashboard enables users to analyze public sentiment and perception by processing YouTube video comments or customized datasets uploaded as CSV or Excel files. Using advanced natural language processing techniques, the dashboard provides sentiment classification, keyword trends, and visual insights to support data-driven decision-making.
        """
    )

    # Data Input Section
    with gr.Tabs() as input_tabs:

        with gr.Tab("🎬 YouTube Video Analysis"):
            with gr.Row():
                video_url = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
                url_load_btn = gr.Button("🎬 Analyze Comments", variant="primary")

        with gr.Tab("📁 File Upload Analysis"):
            with gr.Row():
                file_input = gr.File(label="Upload CSV or Excel File", file_types=[".csv", ".xlsx"])
                file_load_btn = gr.Button("📊 Load & Analyze File", variant="primary")

    # Hidden state component
    comments_df = gr.DataFrame(value=global_df if global_df is not None else generate_default_df(),
                             label="Loaded Comment Data", interactive=False, visible=False)

    # Dashboard Tabs
    with gr.Tabs():
        # Tab 1: Main Analytics Dashboard
        with gr.Tab("Analytics Dashboard"):
            # Summary metrics
            metrics = create_summary_metrics(global_df if global_df is not None else generate_default_df())

            with gr.Row():
                with gr.Column(scale=1):
                    total_comments = gr.Number(value=metrics["total"], label="Total Comments", interactive=False)
                with gr.Column(scale=1):
                    positive_count = gr.Number(value=metrics["positive_pct"], label="Positive %", interactive=False)
                with gr.Column(scale=1):
                    neutral_count = gr.Number(value=metrics["neutral_pct"], label="Neutral %", interactive=False)
                with gr.Column(scale=1):
                    negative_count = gr.Number(value=metrics["negative_pct"], label="Negative %", interactive=False)

            with gr.Row():
                with gr.Column(scale=1):
                    pos_neg_ratio = gr.Number(value=metrics["sentiment_ratio"], label="Positive/Negative Ratio", interactive=False)
                with gr.Column(scale=1):
                    sentiment_trend = gr.Textbox(value=metrics["trend"], label="Sentiment Trend", interactive=False)


            # Visualizations
            gr.Markdown("### 📊 Sentiment Visualizations")

            with gr.Tabs():
                with gr.Tab("Timeline Analysis"):
                    timeline_plot = gr.Plot(value=plot_sentiment_timeline(global_df if global_df is not None else generate_default_df()))

                with gr.Tab("Sentiment Distribution"):
                    distribution_plot = gr.Plot(value=plot_sentiment_distribution(global_df if global_df is not None else generate_default_df()))

                with gr.Tab("Keyword Analysis"):
                    keyword_plot = gr.Plot(value=plot_keyword_analysis(global_df if global_df is not None else generate_default_df()))

            gr.Markdown("### Word Clouds of keyword")

            with gr.Tab("Word Clouds"):
                with gr.Row():
                    sentiment_filter = gr.Dropdown(
                        choices=["All", "Positive", "Neutral", "Negative"],
                        value="All",
                        label="Sentiment Filter"
                    )
                    generate_button = gr.Button("Generate Word Cloud")

                wordcloud_output = gr.Image(label="Word Cloud")

            gr.Markdown("### Data Extracted")
            with gr.Row():
                comments_display = gr.DataFrame(
                    value=global_df if global_df is not None else generate_default_df(),
                    interactive=False
                )

            with gr.Row():
                export_btn = gr.Button("💾 Export & Download CSV", variant="secondary")
            with gr.Row():
                download_component = gr.File(label="Download", visible=True)

        # Tab 2: Quick Analysis
        with gr.Tab("Quick Sentiment Analyzer"):
            gr.Markdown("""
            ### Quick Sentiment Analysis Tool
            Quickly analyze the sentiment of any comment you enter.
            """)

            with gr.Row():
                quick_comment = gr.Textbox(
                    placeholder="Type your comment here...",
                    label="Comment for Analysis",
                    lines=3
                )

            with gr.Row():
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")

            with gr.Row():
                with gr.Column():
                    sentiment_result = gr.Textbox(label="Sentiment")
                with gr.Column():
                    confidence_result = gr.Textbox(label="Confidence")
                with gr.Column():
                    keyword_result = gr.Textbox(label="Key Topics")

        # Tab 3: About & Help
        with gr.Tab("About This Dashboard"):
            gr.Markdown("""
            ## About This Dashboard

            This dashboard allows you to analyze public sentiment from YouTube video comments or uploaded CSV/Excel files.
            It uses natural language processing to detect sentiment, highlight key topics, and reveal emerging trends.
            Whether you are tracking opinions or exploring concerns, the dashboard delivers clear, data-driven insights.

            ### Features:

            - **Multiple Data Sources**: Upload CSV/Excel files or analyze YouTube video comments
            - **Sentiment Analysis**: Automatically classifies comments as Positive, Neutral, or Negative
            - **Keyword Extraction**: Identifies the most important topics in each comment
            - **Time Series Analysis**: Tracks sentiment trends over time
            - **Word Cloud Visualization**: Visual representation of the most common terms
            - **Data Export**: Download collected data for further analysis

            ### How to Use:

            1. Upload a dataset file via the File Upload tab or enter a YouTube URL
            2. View overall sentiment metrics and trends in the Analytics Dashboard
            3. Add new comments using the comment input box
            4. Use the Quick Analyzer for testing sentiment on individual comments
            5. Export data in CSV format for external analysis

            ### File Upload Requirements:

            - CSV or Excel files (.csv, .xlsx)
            - Must contain a 'Text' column with comments
            - Optional 'Datetime' column (will be auto-generated if missing)

            This dashboard is developed by [**Anaclet UKURIKIYEYEZU**](https://portofolio-pi-lac.vercel.app/)
            Feel free to reach out with any questions or feedback!

            ### Contact Information:
             - [**WhatsApp**](https://wa.me/250786698014): +250 786 698 014
             - [**Email**](mailto:anaclet.ukurikiyeyezu@aims.ac.rw): anaclet.ukurikiyeyezu@aims.ac.rw

            """
            )

    # Connect events to functions

    # File upload event
    file_load_btn.click(
        fn=lambda file: load_and_update_all_components(file, None),
        inputs=[file_input],
        outputs=[
            comments_df,  # Hidden state component
            total_comments, positive_count, neutral_count, negative_count,  # Metric displays
            pos_neg_ratio, sentiment_trend,  # Additional metrics
            timeline_plot, distribution_plot, keyword_plot,  # Visualizations
            comments_display  # Comments table
        ]
    )

    # YouTube analysis event
    url_load_btn.click(
        fn=lambda url: load_and_update_all_components(None, url),
        inputs=[video_url],
        outputs=[
            comments_df,  # Hidden state component
            total_comments, positive_count, neutral_count, negative_count,  # Metric displays
            pos_neg_ratio, sentiment_trend,  # Additional metrics
            timeline_plot, distribution_plot, keyword_plot,  # Visualizations
            comments_display  # Comments table
        ]
    )

    # Word cloud generation event
    generate_button.click(
        fn=gradio_generate_wordcloud,
        inputs=[sentiment_filter],
        outputs=[wordcloud_output]
    )

    # Comment analysis event
    analyze_btn.click(
        fn=gradio_analyze_comment,
        inputs=[quick_comment],
        outputs=[sentiment_result, confidence_result, keyword_result]
    )

    # Add comment event
    def add_comment_and_update(comment):
        global global_df
        updated_df, feedback, _ = gradio_add_comment(comment)

        # Update metrics based on the new dataframe
        metrics = create_summary_metrics(updated_df)

        # Return all updated components
        return (
            updated_df,  # Update hidden state
            feedback, "",  # Feedback message and clear input
            metrics["total"], metrics["positive_pct"], metrics["neutral_pct"],  # Update metrics
            metrics["negative_pct"], metrics["sentiment_ratio"], metrics["trend"],
            plot_sentiment_timeline(updated_df),  # Update plots
            plot_sentiment_distribution(updated_df),
            plot_keyword_analysis(updated_df),
            updated_df  # Update display table
        )

    # Export to CSV event
    export_btn.click(
        fn=export_data_to_csv,
        inputs=[comments_display],
        outputs=[download_component]
    )


# # Launch the app
# if __name__ == "__main__":

#     demo.launch(share=True)



if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
        show_tips=False,
        enable_queue=True,
        max_threads=10
    )






