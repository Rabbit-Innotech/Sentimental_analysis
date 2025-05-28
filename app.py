import os
import re
import io
import random
import warnings
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import numpy as np
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from youtube_comment_downloader import YoutubeCommentDownloader

import re
import requests
from datetime import datetime, timedelta
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

class SentimentAnalyzer:
    """Handles sentiment analysis with rule-based approach"""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'best', 'awesome', 'perfect', 'happy', 'pleased',
            'satisfied', 'fair', 'brilliant', 'smart', 'helpful', 'better',
            'improved', 'outstanding', 'superb', 'remarkable', 'impressive',
            'beneficial', 'effective', 'efficient', 'convenient', 'reasonable'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible',
            'disappointing', 'angry', 'frustrated', 'confused', 'expensive',
            'unfair', 'discriminate', 'wrong', 'problem', 'issue', 'difficult',
            'complicated', 'poor', 'fails', 'useless', 'broken', 'slow',
            'unreliable', 'inadequate', 'insufficient', 'disappointing'
        }
        
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'been',
            'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can',
            'not', 'now', 'new', 'way', 'use', 'get', 'make', 'take', 'come',
            'know', 'see', 'think', 'say', 'tell', 'ask', 'give', 'find'
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not text or len(str(text).strip()) == 0:
            return "Neutral", 0.5
        
        text_lower = str(text).lower()
        
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if pos_count > neg_count:
            confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
            return "Positive", confidence
        elif neg_count > pos_count:
            confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
            return "Negative", confidence
        else:
            return "Neutral", 0.5
    
    def extract_keywords(self, text, top_n=3):
        """Extract key words from text"""
        if not text:
            return []
        
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', str(text).lower())
        filtered_words = [w for w in words if w not in self.stop_words and len(w) > 3]
        word_counts = Counter(filtered_words)
        top_keywords = word_counts.most_common(top_n)
        
        return [word for word, count in top_keywords]

class DataProcessor:
    """Handles data processing and management"""
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.data = None
        self.default_comments = [
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
            "Great initiative but poor implementation.",
            "Now I know exactly what I'm paying for. Transparent and fair.",
            "The fare calculator is very helpful.",
            "I've noticed faster service since the new system launched.",
            "Distance-based fares are the future of transportation.",
            "This discriminates against people living in rural areas!",
            "My transportation costs have decreased by 30%!",
            "Very impressed with the new fare calculation technology.",
            "Love how I can now predict exactly what my trip will cost.",
            "Works well in urban areas but rural commuters are suffering."
        ]
    
    def generate_sample_data(self):
        """Generate sample dataset for demo"""
        data = []
        base_time = datetime.now() - timedelta(hours=48)
        
        for i, comment in enumerate(self.default_comments):
            timestamp = base_time + timedelta(hours=random.uniform(0, 48))
            sentiment, score = self.analyzer.analyze_sentiment(comment)
            keywords = self.analyzer.extract_keywords(comment, 3)
            keyword_str = ", ".join(keywords) if keywords else "N/A"
            
            data.append({
                "Datetime": timestamp,
                "Text": comment,
                "Sentiment": sentiment,
                "Score": round(score, 3),
                "Keywords": keyword_str
            })
        
        self.data = pd.DataFrame(data)
        self.data["Datetime"] = pd.to_datetime(self.data["Datetime"])
        return self.data
    
    def process_file(self, file):
        """Process uploaded CSV/Excel file"""
        if file is None:
            return self.generate_sample_data()
        
        try:
            # Read file based on extension
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file.name)
            else:
                return self.generate_sample_data()
            
            # Check for required column
            if 'Text' not in df.columns:
                raise ValueError("File must contain a 'Text' column")
            
            # Process data
            processed_data = []
            for idx, row in df.iterrows():
                text = str(row['Text']) if pd.notna(row['Text']) else ""
                
                # Handle timestamp
                if 'Datetime' in df.columns and pd.notna(row['Datetime']):
                    timestamp = pd.to_datetime(row['Datetime'])
                else:
                    timestamp = datetime.now() - timedelta(hours=len(df)-idx)
                
                # Analyze text
                sentiment, score = self.analyzer.analyze_sentiment(text)
                keywords = self.analyzer.extract_keywords(text, 3)
                keyword_str = ", ".join(keywords) if keywords else "N/A"
                
                processed_data.append({
                    "Datetime": timestamp,
                    "Text": text,
                    "Sentiment": sentiment,
                    "Score": score,
                    "Keywords": keyword_str
                })
            
            self.data = pd.DataFrame(processed_data)
            return self.data
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return self.generate_sample_data()
    
    # def convert_relative_time(self, relative):
    #     """Convert relative time string to datetime"""
    #     now = datetime.now()
    #     try:
    #         match = re.match(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', relative.lower())
    #         if not match:
    #             return now
            
    #         value, unit = int(match.group(1)), match.group(2)
            
    #         time_deltas = {
    #             'second': timedelta(seconds=value),
    #             'minute': timedelta(minutes=value),
    #             'hour': timedelta(hours=value),
    #             'day': timedelta(days=value),
    #             'week': timedelta(weeks=value),
    #             'month': timedelta(days=value * 30),
    #             'year': timedelta(days=value * 365)
    #         }
            
    #         return now - time_deltas.get(unit, timedelta(0))
            
    #     except Exception as e:
    #         print(f"Failed to parse relative time '{relative}': {e}")
    #         return now
    
    # def process_youtube_comments(self, video_url):
    #     """Process YouTube video comments"""
    #     youtube_pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}"
    #     if not re.match(youtube_pattern, video_url):
    #         raise gr.Error("Please provide a valid YouTube video link.")
        
    #     try:
    #         downloader = YoutubeCommentDownloader()
    #         comments = downloader.get_comments_from_url(video_url)
    #         if not comments:
    #             raise gr.Error("No comments found for this video.")
            
    #         processed_data = []
    #         for comment in comments:
    #             text = comment.get('text', '')
    #             timestamp = self.convert_relative_time(comment.get('time', '0 seconds ago'))
                
    #             sentiment, score = self.analyzer.analyze_sentiment(text)
    #             keywords = self.analyzer.extract_keywords(text, 3)
    #             keyword_str = ", ".join(keywords) if keywords else "N/A"
                
    #             processed_data.append({
    #                 "Datetime": timestamp,
    #                 "Text": text,
    #                 "Sentiment": sentiment,
    #                 "Score": score,
    #                 "Keywords": keyword_str
    #             })
            
    #         self.data = pd.DataFrame(processed_data)
    #         self.data["Datetime"] = pd.to_datetime(self.data["Datetime"])
    #         self.data = self.data.sort_values("Datetime").reset_index(drop=True)
    #         return self.data
            
    #     except Exception as e:
    #         raise gr.Error(f"Failed to retrieve comments: {str(e)}")


    
    def convert_relative_time(self, relative):
        """Convert relative time string to datetime"""
        now = datetime.now()
        try:
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
    
    def extract_video_id(self, video_url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'youtube\.com/watch\?v=([^&]+)',
            r'youtu\.be/([^?]+)',
            r'youtube\.com/embed/([^?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
        return None
    
    def process_youtube_comments(self, video_url):
        """Process YouTube video comments using YouTube API"""
        api_key = "AIzaSyDlGn2abWfnPLb5JL2e9H7MrujvEDuBHtI"
        
        # Validate YouTube URL
        youtube_pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}"
        if not re.match(youtube_pattern, video_url):
            raise gr.Error("Please provide a valid YouTube video link.")
        
        # Extract video ID
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise gr.Error("Could not extract video ID from URL.")
        
        try:
            # Fetch comments using YouTube API
            comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'key': api_key,
                'maxResults': 10000,  # Adjust as needed
                'order': 'time'
            }
            
            all_comments = []
            next_page_token = None
            
            while True:
                if next_page_token:
                    params['pageToken'] = next_page_token
                
                response = requests.get(comments_url, params=params)
                
                if response.status_code != 200:
                    raise gr.Error(f"API request failed: {response.status_code} - {response.text}")
                
                data = response.json()
                
                if 'items' not in data or not data['items']:
                    break
                
                for item in data['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'text': comment['textDisplay'],
                        'time': comment['publishedAt'],
                        'author': comment['authorDisplayName'],
                        'likes': comment['likeCount']
                    })
                
                # Check if there are more pages
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
            
            if not all_comments:
                raise gr.Error("No comments found for this video.")
            
            processed_data = []
            for comment in all_comments:
                text = comment.get('text', '')
                # Convert ISO timestamp to datetime
                timestamp = datetime.fromisoformat(comment.get('time', '').replace('Z', '+00:00'))
                
                sentiment, score = self.analyzer.analyze_sentiment(text)
                keywords = self.analyzer.extract_keywords(text, 3)
                keyword_str = ", ".join(keywords) if keywords else "N/A"
                
                processed_data.append({
                    "Datetime": timestamp,
                    "Text": text,
                    "Sentiment": sentiment,
                    "Score": score,
                    "Keywords": keyword_str,
                    "Author": comment.get('author', ''),
                    "Likes": comment.get('likes', 0)
                })
            
            self.data = pd.DataFrame(processed_data)
            self.data["Datetime"] = pd.to_datetime(self.data["Datetime"])
            self.data = self.data.sort_values("Datetime").reset_index(drop=True)
            
            return self.data
            
        except Exception as e:
            raise gr.Error(f"Failed to retrieve comments: {str(e)}")
    
    
    def get_summary_metrics(self):
        """Generate summary metrics"""
        if self.data is None or self.data.empty:
            return {
                "total": 0, "positive": 0, "neutral": 0, "negative": 0,
                "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0,
                "sentiment_ratio": 0.0
            }
        
        total_comments = len(self.data)
        sentiment_counts = self.data["Sentiment"].value_counts().to_dict()
        
        positive = sentiment_counts.get("Positive", 0)
        neutral = sentiment_counts.get("Neutral", 0)
        negative = sentiment_counts.get("Negative", 0)
        
        def pct(count):
            return round((count / total_comments) * 100, 1) if total_comments else 0.0
        
        return {
            "total": total_comments,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "positive_pct": pct(positive),
            "neutral_pct": pct(neutral),
            "negative_pct": pct(negative),
            "sentiment_ratio": round(positive / negative, 2) if negative > 0 else float('inf')
        }
    
    def export_to_csv(self):
        """Export data to CSV file"""
        if self.data is None or self.data.empty:
            return None
        
        try:
            filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_buffer = io.StringIO()
            self.data.to_csv(csv_buffer, index=False)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_buffer.getvalue())
            
            return filename
        except Exception as e:
            print(f"Error exporting data: {e}")
            return None

class Visualizer:
    """Handles all visualization creation"""
    
    def __init__(self, data_processor):
        self.processor = data_processor
        self.colors = {'Positive': '#28a745', 'Neutral': '#17a2b8', 'Negative': '#dc3545'}
    
    def create_sentiment_timeline(self):
        """Create timeline visualization"""
        if self.processor.data is None or self.processor.data.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        df_hour = self.processor.data.copy()
        df_hour['Hour'] = df_hour['Datetime'].dt.floor('H')
        grouped = df_hour.groupby(['Hour', 'Sentiment']).size().reset_index(name='Count')
        
        fig = go.Figure()
        
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            data = grouped[grouped['Sentiment'] == sentiment]
            if not data.empty:
                fig.add_trace(go.Scatter(
                    x=data['Hour'],
                    y=data['Count'],
                    mode='markers+lines',
                    name=sentiment,
                    marker=dict(color=self.colors[sentiment], size=8),
                    line=dict(color=self.colors[sentiment])
                ))
        
        fig.update_layout(
            title="Sentiment Over Time",
            xaxis_title="Time",
            yaxis_title="Number of Comments",
            height=400
        )
        
        return fig
    
    def create_sentiment_distribution(self):
        """Create sentiment distribution charts"""
        if self.processor.data is None or self.processor.data.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        sentiment_counts = self.processor.data['Sentiment'].value_counts()
        colors = [self.colors.get(s, '#6c757d') for s in sentiment_counts.index]
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Distribution", "Counts")
        )
        
        # Pie chart
        fig.add_trace(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=colors
        ), row=1, col=1)
        
        # Bar chart
        fig.add_trace(go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=colors
        ), row=1, col=2)
        
        fig.update_layout(title="Sentiment Distribution", height=400)
        return fig
    
    def create_keyword_analysis(self):
        """Create keyword analysis visualization"""
        if self.processor.data is None or self.processor.data.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Extract all keywords
        all_keywords = []
        for _, row in self.processor.data.iterrows():
            if row['Keywords'] != 'N/A':
                keywords = [k.strip() for k in str(row['Keywords']).split(',')]
                for kw in keywords:
                    if kw:
                        all_keywords.append((kw, row['Sentiment']))
        
        if not all_keywords:
            return go.Figure().add_annotation(text="No keywords available", x=0.5, y=0.5)
        
        # Count keywords by sentiment
        kw_df = pd.DataFrame(all_keywords, columns=['Keyword', 'Sentiment'])
        kw_counts = kw_df.groupby(['Keyword', 'Sentiment']).size().reset_index(name='Count')
        
        # Get top keywords
        top_kw = kw_df['Keyword'].value_counts().head(10).index
        kw_counts = kw_counts[kw_counts['Keyword'].isin(top_kw)]
        
        fig = px.bar(
            kw_counts,
            x='Keyword',
            y='Count',
            color='Sentiment',
            color_discrete_map=self.colors,
            title="Top Keywords by Sentiment"
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_wordcloud(self, sentiment_filter=None):
        """Create word cloud visualization"""
        if self.processor.data is None or self.processor.data.empty:
            return None
        
        # Filter by sentiment if provided
        if sentiment_filter and sentiment_filter != "All":
            filtered_df = self.processor.data[self.processor.data["Sentiment"] == sentiment_filter]
        else:
            filtered_df = self.processor.data
        
        if filtered_df.empty:
            return None
        
        # Combine keywords
        keyword_text = filtered_df["Keywords"].fillna("").str.replace("N/A", "").str.replace(",", " ")
        all_keywords = " ".join(keyword_text)
        
        if not all_keywords.strip():
            return None
        
        # Select colormap based on sentiment
        colormap_map = {
            "Positive": "Greens",
            "Neutral": "Blues", 
            "Negative": "Reds"
        }
        colormap = colormap_map.get(sentiment_filter, "viridis")
        
        # Create word cloud
        wordcloud = WordCloud(
            background_color='white',
            colormap=colormap,
            max_words=50,
            height=500,
        ).generate(all_keywords)
        
        return wordcloud.to_image()

# Initialize global components
data_processor = DataProcessor()
visualizer = Visualizer(data_processor)

# Initialize with sample data
data_processor.generate_sample_data()

def load_and_update_all_components(file, video_url):
    """Load data and update all dashboard components"""
    # Determine which input to process
    if file is not None:
        updated_df = data_processor.process_file(file)
    elif video_url:
        updated_df = data_processor.process_youtube_comments(video_url)
    else:
        updated_df = data_processor.data
    
    # Generate updated metrics and visuals
    metrics = data_processor.get_summary_metrics()
    
    return (
        updated_df,
        metrics["total"], metrics["positive_pct"], metrics["neutral_pct"],
        metrics["negative_pct"], metrics["sentiment_ratio"],
        visualizer.create_sentiment_timeline(),
        visualizer.create_sentiment_distribution(),
        visualizer.create_keyword_analysis(),
        updated_df
    )

def analyze_single_comment(text):
    """Analyze a single comment"""
    if not text:
        return "N/A", 0.0, "N/A"
    
    sentiment, score = data_processor.analyzer.analyze_sentiment(text)
    keywords = data_processor.analyzer.extract_keywords(text, 3)
    keyword_str = ", ".join(keywords) if keywords else "N/A"
    
    return sentiment, score, keyword_str

def generate_wordcloud(sentiment_filter):
    """Generate word cloud with sentiment filter"""
    try:
        filter_value = sentiment_filter if sentiment_filter != "All" else None
        return visualizer.create_wordcloud(filter_value)
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

# Create Gradio interface
def create_interface():
    # Create Gradio Interface
    with gr.Blocks(theme=gr.themes.Soft(), title="Sentiment Analysis Dashboard") as demo:
        gr.Markdown("""
        # 🎯 Smart Dashboard for Analyzing Public Sentiment and Perception
        
        This interactive dashboard enables users to analyze public sentiment and perception by processing 
        YouTube video comments or customized datasets uploaded as CSV or Excel files. Using advanced 
        natural language processing techniques, the dashboard provides sentiment classification, keyword 
        trends, and visual insights to support data-driven decision-making.
        """)
        
        # Data Input Section
        with gr.Tabs() as input_tabs:
            with gr.Tab("🎬 YouTube Video Analysis"):
                with gr.Row():
                    video_url = gr.Textbox(
                        label="YouTube Video URL", 
                        placeholder="https://www.youtube.com/watch?v=..."
                    )
                    url_load_btn = gr.Button("🎬 Analyze Comments", variant="primary")
            
            with gr.Tab("📁 File Upload Analysis"):
                with gr.Row():
                    file_input = gr.File(
                        label="Upload CSV or Excel File", 
                        file_types=[".csv", ".xlsx"]
                    )
                    file_load_btn = gr.Button("📊 Load & Analyze File", variant="primary")
        
        # Hidden state component
        comments_df = gr.DataFrame(
            value=data_processor.data,
            label="Loaded Comment Data", 
            interactive=False, 
            visible=False
        )
        
        # Main Dashboard Tabs
        with gr.Tabs():
            # Analytics Dashboard Tab
            with gr.Tab("📊 Analytics Dashboard"):
                # Summary metrics
                metrics = data_processor.get_summary_metrics()
                
                with gr.Row():
                    total_comments = gr.Number(
                        value=metrics["total"], 
                        label="Total Comments", 
                        interactive=False
                    )
                    positive_count = gr.Number(
                        value=metrics["positive_pct"], 
                        label="Positive %", 
                        interactive=False
                    )
                    neutral_count = gr.Number(
                        value=metrics["neutral_pct"], 
                        label="Neutral %", 
                        interactive=False
                    )
                    negative_count = gr.Number(
                        value=metrics["negative_pct"], 
                        label="Negative %", 
                        interactive=False
                    )
                
                with gr.Row():
                    pos_neg_ratio = gr.Number(
                        value=metrics["sentiment_ratio"], 
                        label="Positive/Negative Ratio", 
                        interactive=False
                    )
                
                # Visualizations
                gr.Markdown("### 📈 Sentiment Visualizations")
                
                with gr.Tabs():
                    with gr.Tab("Timeline Analysis"):
                        timeline_plot = gr.Plot(value=visualizer.create_sentiment_timeline())
                    
                    with gr.Tab("Sentiment Distribution"):
                        distribution_plot = gr.Plot(value=visualizer.create_sentiment_distribution())
                    
                    with gr.Tab("Keyword Analysis"):
                        keyword_plot = gr.Plot(value=visualizer.create_keyword_analysis())
                
                # Word Cloud Section
                gr.Markdown("### ☁️ Word Cloud Visualization")
                
                with gr.Row():
                    sentiment_filter = gr.Dropdown(
                        choices=["All", "Positive", "Neutral", "Negative"],
                        value="All",
                        label="Sentiment Filter"
                    )
                    generate_button = gr.Button("Generate Word Cloud", variant="secondary")
                
                wordcloud_output = gr.Image(label="Word Cloud")
                
                # Data Display
                gr.Markdown("### 📋 Extracted Data")
                comments_display = gr.DataFrame(
                    value=data_processor.data,
                    interactive=False
                )
                
                with gr.Row():
                    export_btn = gr.Button("💾 Export & Download CSV", variant="secondary")
                    download_component = gr.File(label="Download", visible=True)
            
            # Quick Analysis Tab
            with gr.Tab("⚡ Quick Sentiment Analyzer"):
                gr.Markdown("""
                ### Quick Sentiment Analysis Tool
                Quickly analyze the sentiment of any comment you enter.
                """)
                
                quick_comment = gr.Textbox(
                    placeholder="Type your comment here...",
                    label="Comment for Analysis",
                    lines=3
                )
                
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
                
                with gr.Row():
                    sentiment_result = gr.Textbox(label="Sentiment")
                    confidence_result = gr.Textbox(label="Confidence")
                    keyword_result = gr.Textbox(label="Key Topics")
            
            # About Tab
            with gr.Tab("ℹ️ About This Dashboard"):
                gr.Markdown("""
                ## About This Dashboard
                
                This dashboard allows you to analyze public sentiment from YouTube video comments or 
                uploaded CSV/Excel files. It uses natural language processing to detect sentiment, 
                highlight key topics, and reveal emerging trends.
                
                ### 🚀 Features:
                
                - **Multiple Data Sources**: Upload CSV/Excel files or analyze YouTube video comments
                - **Sentiment Analysis**: Automatically classifies comments as Positive, Neutral, or Negative
                - **Keyword Extraction**: Identifies the most important topics in each comment
                - **Time Series Analysis**: Tracks sentiment trends over time
                - **Word Cloud Visualization**: Visual representation of the most common terms
                - **Data Export**: Download collected data for further analysis
                
                ### 📝 How to Use:
                
                1. Upload a dataset file via the File Upload tab or enter a YouTube URL
                2. View overall sentiment metrics and trends in the Analytics Dashboard
                3. Use the Quick Analyzer for testing sentiment on individual comments
                4. Export data in CSV format for external analysis
                
                ### 📋 File Upload Requirements:
                
                - CSV or Excel files (.csv, .xlsx)
                - Must contain a 'Text' column with comments
                - Optional 'Datetime' column (will be auto-generated if missing)
                
                ---
                
                **Developed by [Anaclet UKURIKIYEYEZU](https://portofolio-pi-lac.vercel.app/)**
                
                ### 📞 Contact Information:
                - **WhatsApp**: [+250 786 698 014](https://wa.me/250786698014)
                - **Email**: [anaclet.ukurikiyeyezu@aims.ac.rw](mailto:anaclet.ukurikiyeyezu@aims.ac.rw)
                """)
        
        # Event Handlers
        
        # File upload event
        file_load_btn.click(
            fn=lambda file: load_and_update_all_components(file, None),
            inputs=[file_input],
            outputs=[
                comments_df, total_comments, positive_count, neutral_count, 
                negative_count, pos_neg_ratio, timeline_plot, distribution_plot, 
                keyword_plot, comments_display
            ]
        )
        
        # YouTube analysis event
        url_load_btn.click(
            fn=lambda url: load_and_update_all_components(None, url),
            inputs=[video_url],
            outputs=[
                comments_df, total_comments, positive_count, neutral_count,
                negative_count, pos_neg_ratio, timeline_plot, distribution_plot,
                keyword_plot, comments_display
            ]
        )
        
        # Word cloud generation event
        generate_button.click(
            fn=generate_wordcloud,
            inputs=[sentiment_filter],
            outputs=[wordcloud_output]
        )
        
        # Comment analysis event
        analyze_btn.click(
            fn=analyze_single_comment,
            inputs=[quick_comment],
            outputs=[sentiment_result, confidence_result, keyword_result]
        )
        
        # Export to CSV event
        export_btn.click(
            fn=lambda _: data_processor.export_to_csv(),
            inputs=[comments_display],
            outputs=[download_component]
        )

    
    return demo

# Main execution
if __name__ == "__main__":
    # Get port from environment
    port = int(os.environ.get("PORT", 7860))
    
    # Create and launch app
    app = create_interface()
    
    print(f"Starting server on port {port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=False,
        show_error=True
    )

