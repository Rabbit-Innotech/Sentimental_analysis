import os
import re
import random
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Essential imports only
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

# Disable matplotlib to save memory
os.environ['MPLBACKEND'] = 'Agg'

# Simple fallback sentiment analysis (no heavy models)
def analyze_sentiment(text):
    """Lightweight rule-based sentiment analysis"""
    if not text or len(str(text).strip()) == 0:
        return "Neutral", 0.5
    
    text_lower = str(text).lower()
    
    # Expanded sentiment word lists
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
        'love', 'like', 'best', 'awesome', 'perfect', 'happy', 'pleased', 
        'satisfied', 'fair', 'brilliant', 'smart', 'helpful', 'better', 
        'improved', 'outstanding', 'superb', 'remarkable', 'impressive',
        'beneficial', 'effective', 'efficient', 'convenient', 'reasonable'
    ]
    
    negative_words = [
        'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 
        'disappointing', 'angry', 'frustrated', 'confused', 'expensive', 
        'unfair', 'discriminate', 'wrong', 'problem', 'issue', 'difficult', 
        'complicated', 'poor', 'fails', 'useless', 'broken', 'slow',
        'unreliable', 'inadequate', 'insufficient', 'disappointing'
    ]
    
    # Count sentiment words
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate sentiment
    if pos_count > neg_count:
        confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
        return "Positive", confidence
    elif neg_count > pos_count:
        confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
        return "Negative", confidence
    else:
        return "Neutral", 0.5

def extract_keywords(text, top_n=3):
    """Simple keyword extraction"""
    if not text:
        return []
    
    # Simple word extraction
    words = re.findall(r'\b[a-zA-Z]{3,15}\b', str(text).lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
        'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'been', 
        'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can',
        'not', 'now', 'new', 'way', 'use', 'get', 'make', 'take', 'come',
        'know', 'see', 'think', 'say', 'tell', 'ask', 'give', 'find'
    }
    
    # Filter and count words
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    word_counts = Counter(filtered_words)
    
    # Return top keywords
    top_keywords = word_counts.most_common(top_n)
    return [word for word, count in top_keywords]

# Global dataframe
global_df = None

# Default sample comments
default_comments = [
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

def generate_sample_data():
    """Generate sample dataset"""
    global global_df
    
    data = []
    base_time = datetime.now() - timedelta(hours=48)
    
    for i, comment in enumerate(default_comments):
        # Add some time variation
        timestamp = base_time + timedelta(hours=random.uniform(0, 48))
        
        # Analyze sentiment
        sentiment, score = analyze_sentiment(comment)
        
        # Extract keywords
        keywords = extract_keywords(comment, 3)
        keyword_str = ", ".join(keywords) if keywords else "N/A"
        
        data.append({
            "Datetime": timestamp,
            "Text": comment,
            "Sentiment": sentiment,
            "Score": round(score, 3),
            "Keywords": keyword_str
        })
    
    global_df = pd.DataFrame(data)
    global_df["Datetime"] = pd.to_datetime(global_df["Datetime"])
    return global_df

def process_file(file):
    """Process uploaded CSV/Excel file"""
    global global_df
    
    if file is None:
        return generate_sample_data()
    
    try:
        # Read file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file.name)
        else:
            return generate_sample_data()
        
        # Check for required column
        if 'Text' not in df.columns:
            return generate_sample_data()
        
        # Process data
        processed_data = []
        for idx, row in df.iterrows():
            text = str(row['Text']) if pd.notna(row['Text']) else ""
            
            # Generate timestamp if not provided
            if 'Datetime' in df.columns and pd.notna(row['Datetime']):
                timestamp = pd.to_datetime(row['Datetime'])
            else:
                timestamp = datetime.now() - timedelta(hours=len(df)-idx)
            
            # Analyze
            sentiment, score = analyze_sentiment(text)
            keywords = extract_keywords(text, 3)
            keyword_str = ", ".join(keywords) if keywords else "N/A"
            
            processed_data.append({
                "Datetime": timestamp,
                "Text": text,
                "Sentiment": sentiment,
                "Score": score,
                "Keywords": keyword_str
            })
        
        global_df = pd.DataFrame(processed_data)
        return global_df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return generate_sample_data()

def analyze_single_comment(text):
    """Analyze a single comment"""
    if not text:
        return "N/A", 0.0, "N/A"
    
    sentiment, score = analyze_sentiment(text)
    keywords = extract_keywords(text, 3)
    keyword_str = ", ".join(keywords) if keywords else "N/A"
    
    return sentiment, score, keyword_str

def add_comment_to_data(text, sentiment, score, keywords):
    """Add analyzed comment to dataset"""
    global global_df
    
    if global_df is None:
        generate_sample_data()
    
    new_row = pd.DataFrame([{
        "Datetime": datetime.now(),
        "Text": text,
        "Sentiment": sentiment,
        "Score": score,
        "Keywords": keywords
    }])
    
    global_df = pd.concat([global_df, new_row], ignore_index=True)
    return global_df

# Visualization functions
def create_sentiment_timeline():
    """Create timeline visualization"""
    if global_df is None or global_df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    # Group by hour and sentiment
    df_hour = global_df.copy()
    df_hour['Hour'] = df_hour['Datetime'].dt.floor('H')
    
    grouped = df_hour.groupby(['Hour', 'Sentiment']).size().reset_index(name='Count')
    
    colors = {'Positive': '#28a745', 'Neutral': '#17a2b8', 'Negative': '#dc3545'}
    
    fig = go.Figure()
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        data = grouped[grouped['Sentiment'] == sentiment]
        if not data.empty:
            fig.add_trace(go.Scatter(
                x=data['Hour'],
                y=data['Count'],
                mode='markers+lines',
                name=sentiment,
                marker=dict(color=colors[sentiment], size=8),
                line=dict(color=colors[sentiment])
            ))
    
    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Time",
        yaxis_title="Number of Comments",
        height=400
    )
    
    return fig

def create_sentiment_distribution():
    """Create sentiment distribution charts"""
    if global_df is None or global_df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    sentiment_counts = global_df['Sentiment'].value_counts()
    colors = ['#28a745' if s == 'Positive' else '#17a2b8' if s == 'Neutral' else '#dc3545' 
              for s in sentiment_counts.index]
    
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

def create_keyword_analysis():
    """Create keyword analysis"""
    if global_df is None or global_df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    # Extract all keywords
    all_keywords = []
    for _, row in global_df.iterrows():
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
        color_discrete_map={'Positive': '#28a745', 'Neutral': '#17a2b8', 'Negative': '#dc3545'},
        title="Top Keywords by Sentiment"
    )
    
    fig.update_layout(height=400)
    return fig

# Create Gradio interface
def create_interface():
    # Initialize with sample data
    generate_sample_data()
    
    with gr.Blocks(title="Sentiment Analysis Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 Sentiment Analysis Dashboard")
        gr.Markdown("*Lightweight version optimized for Render deployment*")
        
        with gr.Tab("📁 Upload & Analyze"):
            with gr.Row():
                file_input = gr.File(
                    label="Upload CSV/Excel (must have 'Text' column)",
                    file_types=[".csv", ".xlsx"]
                )
                analyze_btn = gr.Button("📊 Analyze", variant="primary")
            
            df_display = gr.Dataframe(
                value=global_df,
                label="Results",
                interactive=False,
                height=300
            )
        
        with gr.Tab("💬 Single Comment"):
            with gr.Row():
                comment_input = gr.Textbox(
                    label="Enter comment",
                    placeholder="Type your comment here...",
                    lines=3
                )
                analyze_comment_btn = gr.Button("🔍 Analyze", variant="primary")
            
            with gr.Row():
                sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
                score_output = gr.Number(label="Score", interactive=False)
                keywords_output = gr.Textbox(label="Keywords", interactive=False)
            
            add_btn = gr.Button("➕ Add to Dataset", variant="secondary")
        
        with gr.Tab("📈 Charts"):
            refresh_btn = gr.Button("🔄 Refresh Charts", variant="primary")
            
            with gr.Row():
                timeline_plot = gr.Plot(label="Timeline")
                distribution_plot = gr.Plot(label="Distribution")
            
            keyword_plot = gr.Plot(label="Keywords")
        
        # Event handlers
        analyze_btn.click(
            fn=process_file,
            inputs=[file_input],
            outputs=[df_display]
        )
        
        analyze_comment_btn.click(
            fn=analyze_single_comment,
            inputs=[comment_input],
            outputs=[sentiment_output, score_output, keywords_output]
        )
        
        add_btn.click(
            fn=add_comment_to_data,
            inputs=[comment_input, sentiment_output, score_output, keywords_output],
            outputs=[df_display]
        )
        
        refresh_btn.click(
            fn=lambda: (create_sentiment_timeline(), create_sentiment_distribution(), create_keyword_analysis()),
            outputs=[timeline_plot, distribution_plot, keyword_plot]
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

