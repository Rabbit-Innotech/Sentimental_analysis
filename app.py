import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
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
import os
import sys
import gc  # Garbage collection for memory management

# Memory optimization: Import heavy libraries only when needed
def lazy_import_transformers():
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        return None

def lazy_import_keybert():
    try:
        from keybert import KeyBERT
        return KeyBERT
    except ImportError:
        return None

def lazy_import_wordcloud():
    try:
        from wordcloud import WordCloud
        return WordCloud
    except ImportError:
        return None

def lazy_import_youtube_downloader():
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader
        return YoutubeCommentDownloader
    except ImportError:
        return None

# Initialize models with error handling and memory optimization
def initialize_models():
    global classifier, kw_model
    
    try:
        pipeline = lazy_import_transformers()
        if pipeline:
            classifier = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                max_length=512,
                                truncation=True)
        else:
            classifier = None
            
        KeyBERT = lazy_import_keybert()
        if KeyBERT:
            kw_model = KeyBERT()
        else:
            kw_model = None
            
    except Exception as e:
        print(f"Warning: Could not initialize models: {e}")
        classifier = None
        kw_model = None

# Initialize models
classifier = None
kw_model = None

# Label mapping - handling different model outputs
sentiment_map = {
    "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive",
    "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
    "NEGATIVE": "Negative", "NEUTRAL": "Neutral", "POSITIVE": "Positive"
}
color_map = {"Positive": "#2E8B57", "Neutral": "#4682B4", "Negative": "#CD5C5C"}

# Reduced default comments to save memory
comments = [
    "This new distance fare is really fair. I pay less for short trips!",
    "It's confusing, I don't know how much I'll pay now.",
    "RURA should have informed us better about this change.",
    "Good step towards fairness and modernization.",
    "Too expensive now! I hate this new system.",
    "The distance-based system makes so much more sense than flat rates.",
    "Finally a fair system — short-distance commuters benefit the most!",
    "I'm still unsure how the new rates are calculated. Needs clarity.",
    "Smart move toward a fairer system, but more awareness is needed.",
    "I'm paying more now and it feels unjust.",
    "Flat rates were easier to understand, but this is more logical.",
    "Great initiative but poor implementation.",
    "Now I know exactly what I'm paying for. Transparent and fair.",
    "The fare calculator is very helpful.",
    "Distance-based fares are the future of transportation.",
    "I appreciate the transparency but the app needs work.",
    "My transportation costs have decreased by 30%!",
    "We should go back to the old system immediately.",
    "The government did a good job explaining the benefits.",
    "Very impressed with the new fare calculation technology."
]

# Global variable to hold the current dataframe
global_df = None

def convert_relative_time(relative):
    """Convert relative time string to datetime object"""
    now = datetime.now()
    try:
        import re
        match = re.match(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', relative.lower())
        if not match:
            return now

        value, unit = int(match.group(1)), match.group(2)
        
        time_deltas = {
            'second': timedelta(seconds=value),
            'minute': timedelta(minutes=value),
            'hour': timedelta(hours=value),
            'day': timedelta(days=value),
            'week': timedelta(weeks=value),
            'month': timedelta(days=value * 30),
            'year': timedelta(days=value * 365)
        }
        
        return now - time_deltas.get(unit, timedelta(0))
    except Exception as e:
        print(f"Failed to parse relative time '{relative}': {e}")
        return now

def safe_sentiment_analysis(text):
    """Perform sentiment analysis with fallback"""
    if not classifier:
        # Fallback sentiment analysis using simple keyword matching
        text_lower = text.lower()
        positive_words = ['good', 'great', 'excellent', 'fair', 'love', 'amazing', 'perfect', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'wrong', 'unfair', 'expensive']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "Positive", 0.7
        elif neg_count > pos_count:
            return "Negative", 0.7
        else:
            return "Neutral", 0.6
    
    try:
        result = classifier(text[:512])[0]  # Truncate to save memory
        sentiment = sentiment_map.get(result["label"], "Neutral")
        score = round(result["score"], 3)
        return sentiment, score
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return "Neutral", 0.5

def safe_keyword_extraction(text):
    """Extract keywords with fallback"""
    if not kw_model:
        # Simple keyword extraction using common words
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words][:3]
        return ", ".join(keywords) if keywords else "N/A"
    
    try:
        keywords = kw_model.extract_keywords(text, top_n=3)
        return ", ".join([kw[0] for kw in keywords]) if keywords else "N/A"
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return "N/A"

def generate_default_df():
    """Generate default dataset with memory optimization"""
    global global_df
    default_data = []
    start_time = datetime.now() - timedelta(hours=12)  # Reduced time range

    for i, comment in enumerate(comments[:15]):  # Limit to 15 comments to save memory
        timestamp = start_time + timedelta(hours=random.uniform(0, 12))
        
        sentiment, score = safe_sentiment_analysis(comment)
        keyword_str = safe_keyword_extraction(comment)

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
    
    # Force garbage collection
    gc.collect()
    return global_df

def process_uploaded_file(file):
    """Process uploaded file with memory optimization"""
    global global_df

    if file is None:
        global_df = generate_default_df()
        return global_df

    try:
        # Read file with size limits
        if file.name.endswith('.csv'):
            user_df = pd.read_csv(file.name, nrows=1000)  # Limit rows to save memory
        elif file.name.endswith('.xlsx'):
            user_df = pd.read_excel(file.name, nrows=1000)
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel files only.")

        if 'Text' not in user_df.columns:
            raise ValueError("File must contain a 'Text' column with comments.")

        # Handle datetime
        if 'Datetime' not in user_df.columns:
            start_time = datetime.now() - timedelta(hours=len(user_df))
            user_df['Datetime'] = [start_time + timedelta(hours=i) for i in range(len(user_df))]

        user_df = user_df[['Datetime', 'Text']].copy()
        user_df["Datetime"] = pd.to_datetime(user_df["Datetime"])
        user_df["Datetime"] = user_df["Datetime"].dt.floor("1H")
        user_df = user_df.dropna(subset=['Text'])

        # Process in batches to save memory
        sentiments, scores, keywords_list = [], [], []
        
        for text in user_df["Text"]:
            sentiment, score = safe_sentiment_analysis(str(text))
            keyword_str = safe_keyword_extraction(str(text))
            
            sentiments.append(sentiment)
            scores.append(score)
            keywords_list.append(keyword_str)

        user_df["Sentiment"] = sentiments
        user_df["Score"] = scores
        user_df["Keywords"] = keywords_list

        global_df = user_df.sort_values("Datetime").reset_index(drop=True)
        gc.collect()  # Force garbage collection
        return global_df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        global_df = generate_default_df()
        return global_df

def analyze_youtube_comments(video_url):
    """Analyze YouTube comments with memory optimization"""
    YoutubeCommentDownloader = lazy_import_youtube_downloader()
    if not YoutubeCommentDownloader:
        raise gr.Error("YouTube comment downloader not available")
    
    import re
    youtube_pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}"
    if not re.match(youtube_pattern, video_url):
        raise gr.Error("🚫 Please provide a valid YouTube video link.")

    try:
        downloader = YoutubeCommentDownloader()
        comments_iter = downloader.get_comments_from_url(video_url, sort_by=0)
        
        # Limit comments to save memory
        comments_list = []
        for i, comment in enumerate(comments_iter):
            if i >= 100:  # Limit to 100 comments
                break
            comments_list.append(comment)
        
        if not comments_list:
            raise gr.Error("⚠️ No comments found for this video.")
        
        return generate_df(comments_list)
    except Exception as e:
        raise gr.Error(f"❌ Failed to retrieve comments: {str(e)}")

def generate_df(comments):
    """Generate dataframe from comments with memory optimization"""
    global global_df
    default_data = []

    for comment in comments[:50]:  # Limit to 50 comments
        text = comment.get('text', '')[:512]  # Truncate long text
        timestamp = convert_relative_time(comment.get('time', '0 seconds ago'))

        sentiment, score = safe_sentiment_analysis(text)
        keyword_str = safe_keyword_extraction(text)

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
    gc.collect()
    return global_df

def create_wordcloud_simple(df, sentiment_filter=None):
    """Create wordcloud with memory optimization"""
    WordCloud = lazy_import_wordcloud()
    if not WordCloud:
        return None
    
    if df is None or df.empty:
        return None

    try:
        # Filter by sentiment
        if sentiment_filter and sentiment_filter != "All":
            filtered_df = df[df["Sentiment"] == sentiment_filter]
        else:
            filtered_df = df

        if filtered_df.empty:
            return None

        # Combine keywords
        keyword_text = filtered_df["Keywords"].fillna("").str.replace("N/A", "").str.replace(",", " ")
        all_keywords = " ".join(keyword_text)

        if not all_keywords.strip():
            return None

        # Select colormap
        colormap = "viridis"
        if sentiment_filter == "Positive":
            colormap = "Greens"
        elif sentiment_filter == "Neutral":
            colormap = "Blues"
        elif sentiment_filter == "Negative":
            colormap = "Reds"

        # Create wordcloud with reduced size
        wordcloud = WordCloud(
            background_color='white',
            colormap=colormap,
            max_words=30,  # Reduced from 50
            width=400,     # Reduced size
            height=300,
        ).generate(all_keywords)

        return wordcloud.to_image()
    except Exception as e:
        print(f"Error creating wordcloud: {e}")
        return None

def plot_sentiment_timeline(df):
    """Create timeline plot with memory optimization"""
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available", height=400)

    try:
        df_copy = df.copy()
        df_copy["Datetime"] = pd.to_datetime(df_copy["Datetime"])
        df_copy["Time_Bin"] = df_copy["Datetime"].dt.floor("1H")

        grouped = (
            df_copy.groupby(["Time_Bin", "Sentiment"])
            .agg(
                Count=("Text", "count"),
                Score=("Score", "mean")
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
                    marker=dict(size=8, color=color, opacity=0.8),
                    hovertemplate='<b>%{y} comments</b><br>%{x}<extra></extra>'
                )
            )

        fig.update_layout(
            title="Sentiment Over Time",
            height=400,  # Reduced height
            xaxis=dict(title="Time"),
            yaxis=dict(title="Number of Comments"),
            template="plotly_white",
            showlegend=True
        )

        return fig

    except Exception as e:
        print(f"Error in timeline plot: {e}")
        return go.Figure().update_layout(title="Error creating timeline", height=400)

def plot_sentiment_distribution(df):
    """Create distribution plot with memory optimization"""
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available", height=400)

    try:
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Distribution", "Counts")
        )

        # Pie Chart
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts["Sentiment"],
                values=sentiment_counts["Count"],
                textinfo="percent+label",
                marker=dict(colors=[color_map.get(s, "#999999") for s in sentiment_counts["Sentiment"]]),
                hole=0.3
            ),
            row=1, col=1
        )

        # Bar Chart
        fig.add_trace(
            go.Bar(
                x=sentiment_counts["Sentiment"],
                y=sentiment_counts["Count"],
                marker_color=[color_map.get(s, "#999999") for s in sentiment_counts["Sentiment"]]
            ),
            row=1, col=2
        )

        fig.update_layout(
            title="Sentiment Distribution",
            height=400,
            template="plotly_white",
            showlegend=False
        )

        return fig

    except Exception as e:
        print(f"Error in distribution plot: {e}")
        return go.Figure().update_layout(title="Error creating distribution", height=400)

def plot_keyword_analysis(df):
    """Create keyword analysis plot with memory optimization"""
    if df is None or df.empty:
        return go.Figure().update_layout(title="No data available", height=400)

    try:
        all_keywords = []

        for sentiment in ["Positive", "Neutral", "Negative"]:
            sentiment_df = df[df["Sentiment"] == sentiment]
            if sentiment_df.empty:
                continue

            for keywords_str in sentiment_df["Keywords"].dropna():
                if keywords_str and keywords_str.upper() != "N/A":
                    keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                    for kw in keywords:
                        all_keywords.append((kw, sentiment))

        if not all_keywords:
            return go.Figure().update_layout(title="No keyword data", height=400)

        keywords_df = pd.DataFrame(all_keywords, columns=["Keyword", "Sentiment"])
        keyword_counts = keywords_df.groupby(["Keyword", "Sentiment"]).size().reset_index(name="Count")

        # Limit to top 10 keywords
        top_keywords = keywords_df["Keyword"].value_counts().nlargest(10).index
        keyword_counts = keyword_counts[keyword_counts["Keyword"].isin(top_keywords)]

        fig = px.bar(
            keyword_counts,
            x="Keyword",
            y="Count",
            color="Sentiment",
            color_discrete_map=color_map,
            barmode="group",
            title="Top Keywords by Sentiment",
            height=400
        )

        fig.update_layout(template="plotly_white")
        return fig

    except Exception as e:
        print(f"Error in keyword analysis: {e}")
        return go.Figure().update_layout(title="Error creating keywords", height=400)

def create_summary_metrics(df):
    """Create summary metrics with error handling"""
    if df is None or df.empty:
        return {
            "total": 0, "positive": 0, "neutral": 0, "negative": 0,
            "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0,
            "sentiment_ratio": 0.0, "trend": "No data"
        }

    try:
        total_comments = len(df)
        sentiment_counts = df["Sentiment"].value_counts().to_dict()
        
        positive = sentiment_counts.get("Positive", 0)
        neutral = sentiment_counts.get("Neutral", 0)
        negative = sentiment_counts.get("Negative", 0)

        def pct(count):
            return round((count / total_comments) * 100, 1) if total_comments else 0.0

        positive_pct = pct(positive)
        neutral_pct = pct(neutral)
        negative_pct = pct(negative)

        sentiment_ratio = round(positive / negative, 2) if negative > 0 else float('inf')
        trend = "Stable"  # Simplified trend calculation

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
            "sentiment_ratio": 0.0, "trend": "Error"
        }

def analyze_text(comment):
    """Analyze single text with error handling"""
    if not comment or not comment.strip():
        return "N/A", 0, "N/A"

    try:
        sentiment, score = safe_sentiment_analysis(comment)
        keywords = safe_keyword_extraction(comment)
        return sentiment, score, keywords
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return "Error", 0, "Error processing text"

def gradio_analyze_comment(comment):
    """Analyze comment for Gradio interface"""
    try:
        if not comment or not comment.strip():
            return "N/A", "0.0%", "N/A"

        sentiment, score, keywords = analyze_text(comment)
        score_str = f"{score * 100:.1f}%"
        return sentiment, score_str, keywords

    except Exception as e:
        print(f"Error in gradio_analyze_comment: {e}")
        return "Error", "0.0%", "Error processing comment"

def gradio_generate_wordcloud(sentiment_filter):
    """Generate wordcloud for Gradio"""
    try:
        filter_value = sentiment_filter if sentiment_filter != "All" else None
        return create_wordcloud_simple(global_df, filter_value)
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

def export_data_to_csv(df_component):
    """Export data to CSV"""
    global global_df
    try:
        if global_df is not None and not global_df.empty:
            filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            global_df.to_csv(filename, index=False)
            return filename
        return None
    except Exception as e:
        print(f"Error exporting data: {e}")
        return None

def load_and_update_all_components(file, video_url):
    """Load data and update all components"""
    global global_df

    try:
        if file is not None:
            updated_df = process_uploaded_file(file)
        elif video_url:
            updated_df = analyze_youtube_comments(video_url)
        else:
            updated_df = global_df if global_df is not None else generate_default_df()

        metrics = create_summary_metrics(updated_df)
        global_df = updated_df

        return (
            updated_df,
            metrics["total"], metrics["positive_pct"], metrics["neutral_pct"],
            metrics["negative_pct"], metrics["sentiment_ratio"], metrics["trend"],
            plot_sentiment_timeline(updated_df),
            plot_sentiment_distribution(updated_df),
            plot_keyword_analysis(updated_df),
            updated_df
        )
    except Exception as e:
        print(f"Error loading components: {e}")
        default_df = generate_default_df()
        metrics = create_summary_metrics(default_df)
        return (
            default_df,
            metrics["total"], metrics["positive_pct"], metrics["neutral_pct"],
            metrics["negative_pct"], metrics["sentiment_ratio"], metrics["trend"],
            plot_sentiment_timeline(default_df),
            plot_sentiment_distribution(default_df),
            plot_keyword_analysis(default_df),
            default_df
        )

# Initialize models and default data
try:
    initialize_models()
    global_df = generate_default_df()
except Exception as e:
    print(f"Initialization error: {e}")
    global_df = pd.DataFrame()

# Create Gradio interface with memory optimization
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Smart Sentiment Analysis Dashboard
    Analyze public sentiment from YouTube comments or uploaded datasets with AI-powered insights.
    """)

    with gr.Tabs():
        with gr.Tab("🎬 YouTube Analysis"):
            with gr.Row():
                video_url = gr.Textbox(label="YouTube Video URL", 
                                     placeholder="https://www.youtube.com/watch?v=...")
                url_load_btn = gr.Button("🎬 Analyze Comments", variant="primary")

        with gr.Tab("📁 File Upload"):
            with gr.Row():
                file_input = gr.File(label="Upload CSV/Excel File", file_types=[".csv", ".xlsx"])
                file_load_btn = gr.Button("📊 Load & Analyze", variant="primary")

    # Hidden state
    comments_df = gr.DataFrame(value=global_df, label="Data", visible=False)

    with gr.Tabs():
        with gr.Tab("📊 Dashboard"):
            # Metrics
            metrics = create_summary_metrics(global_df)
            
            with gr.Row():
                total_comments = gr.Number(value=metrics["total"], label="Total", interactive=False)
                positive_count = gr.Number(value=metrics["positive_pct"], label="Positive %", interactive=False)
                neutral_count = gr.Number(value=metrics["neutral_pct"], label="Neutral %", interactive=False)
                negative_count = gr.Number(value=metrics["negative_pct"], label="Negative %", interactive=False)

            with gr.Row():
                pos_neg_ratio = gr.Number(value=metrics["sentiment_ratio"], label="Pos/Neg Ratio", interactive=False)
                sentiment_trend = gr.Textbox(value=metrics["trend"], label="Trend", interactive=False)

            # Visualizations
            with gr.Tabs():
                with gr.Tab("Timeline"):
                    timeline_plot = gr.Plot(value=plot_sentiment_timeline(global_df))

                with gr.Tab("Distribution"):
                    distribution_plot = gr.Plot(value=plot_sentiment_distribution(global_df))

                with gr.Tab("Keywords"):
                    keyword_plot = gr.Plot(value=plot_keyword_analysis(global_df))

            # Word Cloud
            gr.Markdown("### Word Cloud")
            with gr.Row():
                sentiment_filter = gr.Dropdown(
                    choices=["All", "Positive", "Neutral", "Negative"],
                    value="All",
                    label="Filter"
                )
                generate_button = gr.Button("Generate")

            wordcloud_output = gr.Image(label="Word Cloud")

            # Data table
            gr.Markdown("### Data")
            comments_display = gr.DataFrame(value=global_df, interactive=False)

            with gr.Row():
                export_btn = gr.Button("💾 Export CSV", variant="secondary")
            download_component = gr.File(label="Download")

        with gr.Tab("🔍 Quick Analyzer"):
            gr.Markdown("### Analyze Any Comment")
            
            quick_comment = gr.Textbox(
                placeholder="Enter your comment here...",
                label="Comment",
                lines=2
            )
            
            analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Row():
                sentiment_result = gr.Textbox(label="Sentiment")
                confidence_result = gr.Textbox(label="Confidence")
                keyword_result = gr.Textbox(label="Keywords")

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Dashboard

            This dashboard analyzes sentiment from text data using AI/ML techniques.

            ### Features:
            - **YouTube Comment Analysis**: Extract and analyze comments from any public video
            - **File Upload**: Process CSV/Excel files with text data
            - **Sentiment Classification**: Automatic positive/neutral/negative classification
            - **Keyword Extraction**: Identify key topics and themes
            - **Visual Analytics**: Interactive charts and word clouds
            - **Data Export**: Download results for further analysis

            ### Usage:
            1. Upload a file or enter a YouTube URL
            2. View sentiment metrics and visualizations
            3. Use Quick Analyzer for individual comments
            4. Export data as needed

            ### File Requirements:
            - CSV or Excel format
            - Must contain 'Text' column
            - Optional 'Datetime' column

            **Developed by Anaclet UKURIKIYEYEZU**
            - WhatsApp: +250 786 698 014
            - Email: anaclet.ukurikiyeyezu@aims.ac.rw
            """)

    # Event handlers
    file_load_btn.click(
        fn=lambda file: load_and_update_all_components(file, None),
        inputs=[file_input],
        outputs=[comments_df, total_comments, positive_count, neutral_count, 
                negative_count, pos_neg_ratio, sentiment_trend,
                timeline_plot, distribution_plot, keyword_plot, comments_display]
    )

    url_load_btn.click(
        fn=lambda url: load_and_update_all_components(None, url),
        inputs=[video_url],
        outputs=[comments_df, total_comments, positive_count, neutral_count,
                negative_count, pos_neg_ratio, sentiment_trend,
                timeline_plot, distribution_plot, keyword_plot, comments_display]
    )

    generate_button.click(
        fn=gradio_generate_wordcloud,
        inputs=[sentiment_filter],
        outputs=[wordcloud_output]
    )

    analyze_btn.click(
        fn=gradio_analyze_comment,
        inputs=[quick_comment],
        outputs=[sentiment_result, confidence_result, keyword_result]
    )

    export_btn.click(
        fn=export_data_to_csv,
        inputs=[comments_display],
        outputs=[download_component]
    )

# Launch configuration for deployment on Render
if __name__ == "__main__":
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 7860))
    
    # Launch the Gradio app
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=port,       # Use the port provided by Render
        share=False,            # Don't create a public tunnel
        debug=False,            # Disable debug mode for production
        show_error=True,        # Show errors in the interface
        quiet=False             # Show startup logs
    )
