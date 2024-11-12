import streamlit as st
import json
import os
import time
import threading
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from test_predictions import run_continuous_predictions, LiveGamePredictor, NBAPredictor
from api_client import EnhancedNBAApiClient

# Enhanced page config
st.set_page_config(
    page_title="NBA Game Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1f2937;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Cards */
    .prediction-card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #3b82f6;
    }
    
    /* Live game indicator */
    .live-game {
        background: linear-gradient(45deg, #ff4b4b, #ff0000);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
        display: inline-block;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Team names */
    .team-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2563eb;
    }
    
    /* Probability bars */
    .probability-bar {
        height: 10px;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 5px;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1f2937;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    
    /* Custom metric styles */
    .custom-metric {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

def create_custom_metric(label, value, delta=None):
    """Create a custom styled metric"""
    metric_html = f"""
        <div class="custom-metric">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {f'<div class="metric-delta">{delta}</div>' if delta else ''}
        </div>
    """
    return st.markdown(metric_html, unsafe_allow_html=True)

def create_team_comparison_chart(home_team, away_team, home_prob, away_prob):
    """Create an interactive team comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[home_team],
        x=[home_prob],
        orientation='h',
        name=home_team,
        marker_color='#3b82f6',
        text=f'{home_prob:.1%}',
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        y=[away_team],
        x=[away_prob],
        orientation='h',
        name=away_team,
        marker_color='#ef4444',
        text=f'{away_prob:.1%}',
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Win Probability Comparison",
        barmode='group',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def display_live_game_card(prediction):
    """Enhanced live game card display"""
    game_info = prediction['game_info']
    pred_info = prediction['prediction']
    
    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        # Header with live indicator
        col1, col2 = st.columns([6,1])
        with col1:
            st.markdown(f"### 🏀 {game_info['home_team']} vs {game_info['away_team']}")
        with col2:
            st.markdown('<span class="live-game">LIVE</span>', unsafe_allow_html=True)
        
        # Score and period
        col1, col2, col3 = st.columns([2,3,2])
        with col1:
            st.markdown(f"""
                <div class="team-name">{game_info['home_team']}</div>
                <div class="score">{game_info['score']['home']}</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style="text-align: center">
                    <div class="period">Period {game_info['period']}</div>
                    <div class="vs">VS</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="team-name">{game_info['away_team']}</div>
                <div class="score">{game_info['score']['away']}</div>
            """, unsafe_allow_html=True)
        
        # Prediction visualization
        st.plotly_chart(
            create_team_comparison_chart(
                game_info['home_team'],
                game_info['away_team'],
                pred_info['win_probability'] if pred_info['predicted_winner'] == game_info['home_team'] else 1-pred_info['win_probability'],
                pred_info['win_probability'] if pred_info['predicted_winner'] == game_info['away_team'] else 1-pred_info['win_probability']
            ),
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)



def display_scheduled_game_card(prediction):
    game_info = prediction['game_info']
    pred_info = prediction['prediction']
    
    col1, col2, col3 = st.columns([2,3,2])
    
    with col1:
        st.markdown("### 🏀 Teams")
        st.write(f"**Home:** {game_info['home_team']}")
        st.write(f"**Away:** {game_info['away_team']}")
        st.write(f"**Start Time:** {game_info['scheduled_start']}")
        
    with col2:
        st.markdown("### 📊 Prediction Details")
        st.write(f"**Predicted Winner:** {pred_info['predicted_winner']}")
        st.write(f"**Win Probability:** {pred_info['win_probability']:.1%}")
        st.write(f"**Confidence:** {pred_info['confidence_level']}")
        
    with col3:
        st.markdown("### 📈 Model Agreement")
        total_models = len(pred_info['model_predictions'])
        agreement = sum(1 for model in pred_info['model_predictions'].values() 
                       if model['predicted_winner'] == pred_info['predicted_winner'])
        st.write(f"**Models in Agreement:** {agreement}/{total_models}")
    
    with st.expander("View Detailed Model Predictions"):
        for model, pred in pred_info['model_predictions'].items():
            st.write(f"**{model}:**")
            st.write(f"- Winner: {pred['predicted_winner']}")
            st.write(f"- Probability: {pred['win_probability']}")

def clean_old_predictions():
    """Delete old prediction files"""
    directories = ["predictions/scheduled", "predictions/live"]
    for directory in directories:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    # Delete file if it's older than 5 minutes
                    if time.time() - os.path.getmtime(file_path) > 300:  # 300 seconds = 5 minutes
                        os.remove(file_path)
                except Exception as e:
                    st.error(f"Error deleting old prediction file {file}: {str(e)}")

def auto_refresh():
    while True:
        time.sleep(300)  # Wait for 5 minutes instead of 1 minute
        st.experimental_rerun()

def load_predictions(include_live=True):
    """Load both scheduled and live game predictions from the respective directories"""
    predictions = []
    current_time = time.time()
    
    # Load scheduled game predictions
    scheduled_dir = "predictions/scheduled"
    if os.path.exists(scheduled_dir):
        for file in os.listdir(scheduled_dir):
            if file.endswith(".json"):
                file_path = os.path.join(scheduled_dir, file)
                # Only load files less than 5 minutes old
                if current_time - os.path.getmtime(file_path) <= 300:
                    try:
                        with open(file_path, 'r') as f:
                            pred = json.load(f)
                            pred['is_live'] = False
                            predictions.append(pred)
                    except json.JSONDecodeError:
                        st.warning(f"Error loading prediction file: {file}")
                    except Exception as e:
                        st.error(f"Unexpected error loading {file}: {str(e)}")
    
    # Load live game predictions
    if include_live:
        live_dir = "predictions/live"
        if os.path.exists(live_dir):
            for file in os.listdir(live_dir):
                if file.endswith(".json"):
                    file_path = os.path.join(live_dir, file)
                    # Only load files less than 5 minutes old
                    if current_time - os.path.getmtime(file_path) <= 300:
                        try:
                            with open(file_path, 'r') as f:
                                pred = json.load(f)
                                pred['is_live'] = True
                                predictions.append(pred)
                        except json.JSONDecodeError:
                            st.warning(f"Error loading live prediction file: {file}")
                        except Exception as e:
                            st.error(f"Unexpected error loading {file}: {str(e)}")
    
    # Sort predictions by date/time
    predictions.sort(key=lambda x: x['game_info'].get('scheduled_start', ''))
    
    return predictions


def main():
    st.title("🏀 NBA Game Predictions Dashboard")
    
    # Start auto-refresh thread
    if 'auto_refresh_started' not in st.session_state:
        st.session_state.auto_refresh_started = True
        threading.Thread(target=auto_refresh, daemon=True).start()
    
    # Sidebar
    st.sidebar.title("Controls")
    auto_update = st.sidebar.checkbox("Auto Update (5 min)", value=True)
    manual_update = st.sidebar.button("Manual Update")
    
    # Clean old predictions before loading new ones
    clean_old_predictions()
    
    if manual_update or (auto_update and time.time() % 300 < 1):
        with st.spinner("Updating predictions..."):
            api_client = EnhancedNBAApiClient('89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c')
            base_predictor = NBAPredictor('saved_models')
            live_predictor = LiveGamePredictor(base_predictor)
            run_continuous_predictions()
    
    # Load predictions
    all_predictions = load_predictions()
    live_games = [p for p in all_predictions if p.get('is_live', False)]
    scheduled_games = [p for p in all_predictions if not p.get('is_live', False)]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Live Games", len(live_games))
    with col2:
        st.metric("Scheduled Games", len(scheduled_games))
    with col3:
        st.metric("High Confidence Predictions", 
                 sum(1 for p in scheduled_games if p['prediction']['confidence_level'] == 'High'))
    with col4:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    # Live Games Section
    st.markdown("## 🔴 Live Games")
    if live_games:
        for prediction in live_games:
            with st.container():
                st.markdown("---")
                display_live_game_card(prediction)
    else:
        st.info("No live games at the moment")
    
    # Scheduled Games Section
    st.markdown("## 📅 Scheduled Games")
    if scheduled_games:
        # Filter options
        st.markdown("### 🔍 Filter Predictions")
        col1, col2 = st.columns(2)
        with col1:
            confidence_filter = st.multiselect(
                "Confidence Level",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Start Time", "Confidence", "Win Probability"]
            )
        
        # Sort predictions
        if sort_by == "Start Time":
            scheduled_games.sort(key=lambda x: x['game_info']['scheduled_start'])
        elif sort_by == "Confidence":
            confidence_map = {"High": 3, "Medium": 2, "Low": 1}
            scheduled_games.sort(key=lambda x: confidence_map[x['prediction']['confidence_level']], 
                               reverse=True)
        else:
            scheduled_games.sort(key=lambda x: x['prediction']['win_probability'], 
                               reverse=True)
        
        # Display filtered predictions
        for prediction in scheduled_games:
            if prediction['prediction']['confidence_level'] in confidence_filter:
                with st.container():
                    st.markdown("---")
                    display_scheduled_game_card(prediction)
    else:
        st.info("No scheduled games found")
    
    # Additional Analysis Section
    if scheduled_games or live_games:
        st.markdown("## 📊 Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Performance")
            model_stats = {}
            for pred in scheduled_games + live_games:
                for model, model_pred in pred['prediction'].get('model_predictions', {}).items():
                    if model not in model_stats:
                        model_stats[model] = {'total': 0, 'agreement': 0}
                    model_stats[model]['total'] += 1
                    if model_pred['predicted_winner'] == pred['prediction']['predicted_winner']:
                        model_stats[model]['agreement'] += 1
            
            # Display model agreement stats
            for model, stats in model_stats.items():
                agreement_rate = stats['agreement'] / stats['total'] if stats['total'] > 0 else 0
                st.write(f"**{model}** agreement rate: {agreement_rate:.1%}")
        
        with col2:
            st.markdown("### Prediction Distribution")
            # Create histogram of win probabilities
            import plotly.express as px
            probs = [p['prediction']['win_probability'] for p in scheduled_games]
            if probs:
                fig = px.histogram(
                    probs, 
                    title="Distribution of Win Probabilities",
                    labels={'value': 'Win Probability', 'count': 'Number of Games'},
                    nbins=20
                )
                st.plotly_chart(fig)
    
    # Footer with last update time
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Update auto-refresh JavaScript to 5 minutes
    if auto_update:
        st.markdown(
            """
            <script>
                var timeout = setTimeout(function() {
                    window.location.reload();
                }, 300000);  // 300000 ms = 5 minutes
            </script>
            """,
            unsafe_allow_html=True
        )

def initialize_session_state():
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'update_count' not in st.session_state:
        st.session_state.update_count = 0

if __name__ == "__main__":
    initialize_session_state()
    main()
