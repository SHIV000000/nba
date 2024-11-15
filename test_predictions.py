# 1. Imports
import streamlit as st
import json
import os
import time
import threading
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from test_predictions import run_continuous_predictions, LiveGamePredictor, NBAPredictor
from api_client import EnhancedNBAApiClient
import atexit
import logging
from typing import Dict

# 2. Page config and CSS
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
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Add hover effect */
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
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
    
    /* Prediction card sections */
    .score-prediction {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .key-factors {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Prediction sections */
    .prediction-section {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Stats table */
    .stats-table {
        width: 100%;
        margin: 1rem 0;
    }
    
    .stats-table td {
        padding: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Model predictions */
    .model-prediction {
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
    
    /* Confidence indicators */
    .confidence-high {
        color: #15803d;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #b45309;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc2626;
        font-weight: bold;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1e40af;
        margin: 1rem 0 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Helper functions
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

def create_metric(label, value):
    """Create a custom styled metric"""
    return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """

def auto_update():
    """Function to handle automatic updates"""
    current_time = time.time()
    
    # Initialize last update time if not exists
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = current_time
        return False
    
    # Check if 5 minutes have passed
    time_elapsed = current_time - st.session_state.last_update_time
    if time_elapsed >= 300:  # 5 minutes in seconds
        try:
            logging.info("Starting auto-update...")
            st.session_state.is_predicting = True
            clean_old_predictions()
            run_continuous_predictions(timeout_minutes=3)
            st.session_state.last_update_time = current_time
            st.session_state.last_prediction_time = current_time
            st.session_state.update_counter += 1
            st.session_state.is_predicting = False
            logging.info("Auto-update completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error in auto-update: {str(e)}")
            st.session_state.is_predicting = False
            return False
    return False

def show_update_status():
    """Show update status in sidebar with countdown timer"""
    if not hasattr(st.session_state, 'last_update_time'):
        return
        
    current_time = time.time()
    time_since_last_update = current_time - st.session_state.last_update_time
    time_until_next_update = max(300 - time_since_last_update, 0)
    
    minutes = int(time_until_next_update // 60)
    seconds = int(time_until_next_update % 60)
    
    st.sidebar.markdown("### Update Status")
    
    # Show last update time
    last_update = datetime.fromtimestamp(st.session_state.last_update_time)
    st.sidebar.info(f"Last update: {last_update.strftime('%H:%M:%S')}")
    
    # Show countdown
    if time_until_next_update > 0:
        st.sidebar.warning(f"Next update in: {minutes:02d}:{seconds:02d}")
    else:
        st.sidebar.success("Update due...")
    
    # Show prediction status
    if st.session_state.is_predicting:
        st.sidebar.warning("⏳ Predictions in progress...")
    else:
        st.sidebar.success("✅ Ready for next update")

# 3. Add Data Validation Function
def validate_prediction_data(prediction):
    """Validate prediction data structure and content"""
    try:
        # Check basic structure
        required_fields = ['game_info', 'prediction', 'timestamp']
        if not all(field in prediction for field in required_fields):
            return False

        # Validate game info
        game_info = prediction['game_info']
        required_game_fields = ['id', 'home_team', 'away_team', 'scheduled_start']
        if not all(field in game_info for field in required_game_fields):
            return False

        # Validate timestamp
        timestamp = datetime.fromisoformat(prediction['timestamp'].replace('Z', '+00:00'))
        current_time = datetime.now(timestamp.tzinfo)
        
        # Check if prediction is not too old (more than 6 hours)
        if (current_time - timestamp).total_seconds() > 21600:
            return False

        return True
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return False

# 4. Modify Load Predictions Function
def load_predictions(include_live=True):
    """Load only valid and recent predictions"""
    predictions = []
    current_time = time.time()
    
    def get_latest_prediction_files(directory):
        game_files = {}
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.json'):
                    game_id = file.split('_')[1]
                    file_path = os.path.join(directory, file)
                    mod_time = os.path.getmtime(file_path)
                    
                    # Only include files less than 6 hours old
                    if current_time - mod_time <= 21600:  # 6 hours in seconds
                        if game_id not in game_files or mod_time > game_files[game_id][1]:
                            game_files[game_id] = (file_path, mod_time)
        
        return [path for path, _ in game_files.values()]

    # Load and validate predictions
    for directory in ["predictions/scheduled", "predictions/live"]:
        for file_path in get_latest_prediction_files(directory):
            try:
                with open(file_path, 'r') as f:
                    pred = json.load(f)
                    
                    if validate_prediction_data(pred):
                        pred['is_live'] = 'live' in directory
                        predictions.append(pred)
                    else:
                        logging.warning(f"Invalid prediction data in {file_path}")
                        # Optionally remove invalid files
                        os.remove(file_path)
                        
            except Exception as e:
                logging.error(f"Error loading prediction file {file_path}: {str(e)}")

    # Sort predictions by scheduled start time
    predictions.sort(key=lambda x: x['game_info']['scheduled_start'])
    return predictions

# 5. Add Update Control Function
def should_update_predictions():
    """Determine if predictions should be updated"""
    if 'last_update_time' not in st.session_state:
        return True
        
    current_time = time.time()
    time_since_update = current_time - st.session_state.last_update_time
    
    # Check if any games are live
    predictions = load_predictions()
    has_live_games = any(p.get('is_live', False) for p in predictions)
    
    # Update more frequently if there are live games
    update_interval = 180 if has_live_games else 300  # 3 or 5 minutes
    
    return time_since_update >= update_interval

# 6. Modify Main Update Logic
def update_predictions():
    """Handle prediction updates with proper control"""
    try:
        if not should_update_predictions():
            return False
            
        logging.info("Starting prediction update...")
        
        # Clean old predictions first
        clean_old_predictions()
        
        # Run predictions with timeout
        success = run_continuous_predictions(timeout_minutes=3)
        
        if success:
            st.session_state.last_update_time = time.time()
            st.session_state.update_counter += 1
            logging.info("Prediction update completed successfully")
            return True
        else:
            logging.error("Prediction update failed")
            return False
            
    except Exception as e:
        logging.error(f"Error in update_predictions: {str(e)}")
        return False

# 4. Core functionality
def display_live_game_card(prediction, key_prefix=None):
    """Enhanced live game card display with unique keys"""
    game_info = prediction['game_info']
    pred_info = prediction['prediction']
    
    # Generate unique key for this game card
    unique_key = f"{key_prefix}_{game_info['id']}" if key_prefix else game_info['id']
    
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
            st.container().markdown(f"""
                <div class="team-name">{game_info['home_team']}</div>
                <div class="score">{game_info['score']['home']}</div>
            """, unsafe_allow_html=True)
        with col2:
            st.container().markdown(f"""
                <div style="text-align: center">
                    <div class="period">Period {game_info['period']}</div>
                    <div class="vs">VS</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.container().markdown(f"""
                <div class="team-name">{game_info['away_team']}</div>
                <div class="score">{game_info['score']['away']}</div>
            """, unsafe_allow_html=True)
        
        # Prediction visualization
        home_prob = pred_info['win_probability'] if pred_info['predicted_winner'] == game_info['home_team'] else (1 - pred_info['win_probability'])
        away_prob = pred_info['win_probability'] if pred_info['predicted_winner'] == game_info['away_team'] else (1 - pred_info['win_probability'])
        
        # Create chart with unique key
        chart = create_team_comparison_chart(
            game_info['home_team'],
            game_info['away_team'],
            home_prob,
            away_prob
        )
        
        st.plotly_chart(chart, use_container_width=True, key=f"chart_{unique_key}")
        
        # Additional prediction details
        st.markdown('</div>', unsafe_allow_html=True)

def display_scheduled_game_card(prediction, key_prefix=None):
    """Display comprehensive scheduled game prediction card."""
    try:
        game_info = prediction.get('game_info', {})
        pred_info = prediction.get('prediction', {})
        
        if not game_info or not pred_info:
            logging.warning("Invalid prediction structure")
            return
        
        # Extract win probability and predicted winner directly
        win_prob = float(pred_info.get('win_probability', 0.5))
        predicted_winner = pred_info.get('predicted_winner', '')
        
        # Calculate home and away probabilities
        home_prob = win_prob if predicted_winner == game_info.get('home_team') else 1 - win_prob
        away_prob = 1 - home_prob
        
        with st.container():
            st.markdown(f"<div class='prediction-card'>", unsafe_allow_html=True)
            
            # Matchup Section
            st.markdown("<div class='section-header'>🏀 MATCHUP:</div>", unsafe_allow_html=True)
            st.markdown(f"{game_info['home_team']} (Home) vs {game_info['away_team']} (Away)")
            
            # Win Probability Section
            st.markdown("<div class='section-header'>📊 WIN PROBABILITY:</div>", unsafe_allow_html=True)
            st.markdown(f"{game_info['home_team']}: {home_prob:.1%}")
            st.markdown(f"{game_info['away_team']}: {away_prob:.1%}")
            
            # Score prediction section
            score_pred = pred_info.get('score_prediction', {})
            if score_pred:
                st.markdown("<div class='section-header'>🎯 PREDICTED SCORE RANGE:</div>", unsafe_allow_html=True)
                st.markdown(f"{game_info.get('home_team')}: {score_pred.get('home_low', 0)}-{score_pred.get('home_high', 0)} points")
                st.markdown(f"{game_info.get('away_team')}: {score_pred.get('away_low', 0)}-{score_pred.get('away_high', 0)} points")
            
            # Summary section
            st.markdown("<div class='section-header'>🏆 PREDICTION SUMMARY:</div>", unsafe_allow_html=True)
            st.markdown(f"Predicted Winner: {predicted_winner}")
            st.markdown(f"Win Confidence: {win_prob:.1%}")
            st.markdown(f"Confidence Level: {pred_info.get('confidence_level', 'Medium')}")
            
            # Model predictions section
            model_preds = pred_info.get('model_predictions', {})
            if isinstance(model_preds, dict):  # Check if it's a dictionary
                st.markdown("<div class='section-header'>🤖 MODEL PREDICTIONS:</div>", unsafe_allow_html=True)
                for model, pred in model_preds.items():
                    if isinstance(pred, dict):
                        model_prob = float(pred.get('win_probability', 0.5))
                        model_winner = pred.get('predicted_winner', '')
                    else:
                        model_prob = float(pred) if isinstance(pred, (int, float)) else 0.5
                        model_winner = predicted_winner
                    
                    confidence = 'High' if model_prob > 0.65 else 'Medium' if model_prob > 0.55 else 'Low'
                    st.markdown(f"- {model}: {model_winner} ({model_prob:.1%}) - {confidence} Confidence")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        logging.error(f"Error displaying prediction card: {str(e)}")
        st.error("Error displaying prediction details")

def clean_old_predictions():
    """Delete old prediction files and keep only the latest for each game"""
    directories = ["predictions/scheduled", "predictions/live"]
    for directory in directories:
        if os.path.exists(directory):
            # Group files by game ID
            game_files = {}
            for file in os.listdir(directory):
                if file.endswith(".json"):
                    game_id = file.split('_')[1]  # Extract game ID from filename
                    file_path = os.path.join(directory, file)
                    if game_id not in game_files:
                        game_files[game_id] = []
                    game_files[game_id].append((file_path, os.path.getmtime(file_path)))
            
            # Keep only the latest file for each game
            for game_id, files in game_files.items():
                # Sort files by modification time
                sorted_files = sorted(files, key=lambda x: x[1])
                # Remove all but the latest file
                for file_path, _ in sorted_files[:-1]:
                    try:
                        os.remove(file_path)
                        logging.debug(f"Removed old prediction file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error deleting file {file_path}: {str(e)}")

def display_predictions(predictions, key=None):
    """Display predictions with custom metrics"""
    # Filter out invalid predictions
    valid_predictions = []
    for p in predictions:
        if validate_prediction_structure(p):
            valid_predictions.append(p)
        else:
            logging.warning(f"Invalid prediction structure for game {p.get('game_info', {}).get('id', 'unknown')}")
    
    live_games = [p for p in valid_predictions if p.get('is_live', False)]
    scheduled_games = [p for p in valid_predictions if not p.get('is_live', False)]
    
    # Display metrics using custom containers
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric("Live Games", len(live_games)), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric("Scheduled Games", len(scheduled_games)), unsafe_allow_html=True)
    with col3:
        high_confidence = sum(1 for p in scheduled_games 
                            if p['prediction'].get('confidence_level') == 'High')
        st.markdown(create_metric("High Confidence", high_confidence), unsafe_allow_html=True)
    with col4:
        last_update = datetime.fromtimestamp(st.session_state.last_prediction_time).strftime("%H:%M:%S")
        st.markdown(create_metric("Last Update", last_update), unsafe_allow_html=True)
    
    # Display games
    if live_games:
        st.markdown("## 🔴 Live Games")
        for i, game in enumerate(live_games):
            display_live_game_card(game, key_prefix=f"live_{key}_{i}")
    
    if scheduled_games:
        st.markdown("## 📅 Scheduled Games")
        for i, game in enumerate(scheduled_games):
            display_scheduled_game_card(game, key_prefix=f"scheduled_{key}_{i}")

def show_prediction_status():
    """Show prediction service status"""
    if st.session_state.is_predicting:
        st.sidebar.warning("⏳ Prediction service is running...")
    else:
        last_update = datetime.fromtimestamp(st.session_state.last_prediction_time)
        st.sidebar.info(f"✅ Last prediction: {last_update.strftime('%H:%M:%S')}")

def cleanup():
    """Clean up resources when the app stops"""
    if hasattr(st.session_state, 'auto_refresh_thread') and st.session_state.auto_refresh_thread:
        st.session_state.auto_refresh_thread = None
    clean_old_predictions()

atexit.register(cleanup)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = time.time()
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()
    if 'is_predicting' not in st.session_state:
        st.session_state.is_predicting = False
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    if 'auto_update_enabled' not in st.session_state:
        st.session_state.auto_update_enabled = True

def show_auto_refresh_status():
    """Show auto-refresh status in sidebar"""
    if st.session_state.auto_refresh_thread is not None:
        st.sidebar.success("🔄 Auto-refresh is active")
        next_update = datetime.fromtimestamp(st.session_state.last_prediction_time + 300)
        st.sidebar.info(f"Next update at: {next_update.strftime('%H:%M:%S')}")

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def log_update_cycle():
    """Log update cycle information"""
    current_time = time.time()
    time_since_last_update = current_time - st.session_state.last_update_time
    
    logging.info(f"""
    Update Cycle Status:
    - Current Time: {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}
    - Last Update: {datetime.fromtimestamp(st.session_state.last_update_time).strftime('%H:%M:%S')}
    - Time Since Last Update: {time_since_last_update:.1f} seconds
    - Auto-Update Enabled: {st.session_state.get('auto_update_enabled', False)}
    - Is Predicting: {st.session_state.get('is_predicting', False)}
    """)

def create_timer():
    """Create a hidden timer component that triggers updates"""
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
        st.session_state.iterations = 0

    # Update every second
    placeholder = st.empty()
    current_time = time.time()
    elapsed = int(current_time - st.session_state.start_time)
    
    if elapsed >= 300:  # 5 minutes
        st.session_state.start_time = current_time
        st.session_state.iterations += 1
        return True
    
    return False

# 7. Update Main Function
def main():
    initialize_session_state()
    
    st.title("🏀 NBA Game Predictions Dashboard")
    
    # Sidebar controls
    st.sidebar.title("Controls")
    auto_update = st.sidebar.checkbox("Enable Auto-Update", value=True)
    
    # Manual update button
    if st.sidebar.button("Update Now"):
        with st.spinner("Updating predictions..."):
            if update_predictions():
                st.success("Update completed!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Update failed. Please try again.")
    
    # Auto-update logic
    if auto_update and should_update_predictions():
        with st.spinner("Running scheduled update..."):
            if update_predictions():
                st.rerun()
    
    # Load and display predictions
    predictions = load_predictions()
    
    if not predictions:
        st.warning("No valid predictions available. Please update predictions.")
        return
    
    # Display predictions
    display_predictions(predictions, key=st.session_state.update_counter)
    
    # Show update status
    show_update_status()
    
    # Rerun for countdown update
    if auto_update:
        time.sleep(1)
        st.rerun()

def show_update_status():
    """Show update status with accurate countdown"""
    if 'start_time' not in st.session_state:
        return
        
    current_time = time.time()
    elapsed = current_time - st.session_state.start_time
    time_remaining = max(300 - elapsed, 0)
    
    minutes = int(time_remaining // 60)
    seconds = int(time_remaining % 60)
    
    st.sidebar.markdown("### Update Status")
    
    # Show last update time
    if 'last_prediction_time' in st.session_state:
        last_update = datetime.fromtimestamp(st.session_state.last_prediction_time)
        st.sidebar.info(f"Last update: {last_update.strftime('%H:%M:%S')}")
    
    # Show countdown
    if time_remaining > 0:
        st.sidebar.warning(f"Next update in: {minutes:02d}:{seconds:02d}")
        
        # Add a progress bar
        progress = 1 - (time_remaining / 300)
        st.sidebar.progress(progress)
    else:
        st.sidebar.success("Update due...")

# 4. Update Session State Initialization
def initialize_session_state():
    """Initialize all session state variables"""
    current_time = time.time()
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = current_time
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = current_time
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    if 'iterations' not in st.session_state:
        st.session_state.iterations = 0

# Add a Heartbeat Function
def heartbeat():
    """Create a heartbeat to ensure continuous updates"""
    if 'heartbeat' not in st.session_state:
        st.session_state.heartbeat = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.heartbeat >= 1:
        st.session_state.heartbeat = current_time
        return True
    return False

def run_predictions_with_retry():
    """Run predictions with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Generating predictions (attempt {attempt + 1}/{max_retries})..."):
                run_continuous_predictions(timeout_minutes=3)
                return True
        except Exception as e:
            logging.error(f"Prediction attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                st.error("Failed to generate predictions. Please try again later.")
                return False

def validate_prediction_structure(prediction):
    """Validate the prediction structure including score prediction"""
    try:
        # Check basic structure
        if not all(key in prediction for key in ['game_info', 'prediction']):
            logging.error(f"Missing top-level keys. Found: {list(prediction.keys())}")
            return False
            
        # Validate game_info
        game_info = prediction['game_info']
        required_game_fields = ['id', 'home_team', 'away_team']
        if not all(field in game_info for field in required_game_fields):
            logging.error(f"Missing game_info fields. Found: {list(game_info.keys())}")
            return False
            
        # Validate prediction structure
        pred_info = prediction['prediction']
        required_pred_fields = ['predicted_winner', 'win_probability', 'confidence_level']
        if not all(field in pred_info for field in required_pred_fields):
            logging.error(f"Missing prediction fields. Found: {list(pred_info.keys())}")
            return False
            
        # Validate score prediction
        pred_info = prediction['prediction']
        if 'score_prediction' in pred_info:
            score_pred = pred_info['score_prediction']
            required_score_fields = ['home_low', 'home_high', 'away_low', 'away_high']
            if not all(field in score_pred for field in required_score_fields):
                logging.error(f"Invalid score prediction structure")
                return False
                
        return True
        
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return False

# 9. Continue Statistics Display
def display_team_stats(team_info: Dict):
    """Display team statistics with charts."""
    stats = team_info['stats']['statistics'][0]
    
    # Create radar chart for key stats
    fig = go.Figure(data=go.Scatterpolar(
        r=[
            stats.get('points', 0),
            stats.get('fieldGoalsPercentage', 0),
            stats.get('threePointsPercentage', 0),
            stats.get('reboundsTotal', 0),
            stats.get('assists', 0),
            stats.get('steals', 0),
            stats.get('blocks', 0)
        ],
        theta=['Points', 'FG%', '3P%', 'Rebounds', 'Assists', 'Steals', 'Blocks'],
        fill='toself',
        name=team_info['name']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed stats table
    st.markdown("#### Detailed Statistics")
    stats_df = pd.DataFrame({
        'Metric': [
            'Points per Game',
            'Field Goal %',
            '3-Point %',
            'Free Throw %',
            'Rebounds',
            'Assists',
            'Steals',
            'Blocks',
            'Turnovers'
        ],
        'Value': [
            f"{stats.get('points', 0):.1f}",
            f"{stats.get('fieldGoalsPercentage', 0):.1f}%",
            f"{stats.get('threePointsPercentage', 0):.1f}%",
            f"{stats.get('freeThrowsPercentage', 0):.1f}%",
            f"{stats.get('reboundsTotal', 0):.1f}",
            f"{stats.get('assists', 0):.1f}",
            f"{stats.get('steals', 0):.1f}",
            f"{stats.get('blocks', 0):.1f}",
            f"{stats.get('turnovers', 0):.1f}"
        ]
    })
    st.table(stats_df)

def display_injury_report(prediction: Dict):
    """Display comprehensive injury report."""
    st.markdown("### 🏥 Injury Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{prediction['game_info']['home_team']['name']} Injuries")
        if prediction['game_info']['home_team']['injuries']:
            injury_df = pd.DataFrame(prediction['game_info']['home_team']['injuries'])
            st.dataframe(injury_df, hide_index=True)
        else:
            st.info("No reported injuries")
            
    with col2:
        st.subheader(f"{prediction['game_info']['away_team']['name']} Injuries")
        if prediction['game_info']['away_team']['injuries']:
            injury_df = pd.DataFrame(prediction['game_info']['away_team']['injuries'])
            st.dataframe(injury_df, hide_index=True)
        else:
            st.info("No reported injuries")

def display_model_analysis(prediction: Dict):
    """Display detailed model analysis and predictions."""
    st.markdown("### 🤖 Model Analysis")
    
    # Model predictions chart
    model_data = []
    for model, pred in prediction['model_predictions'].items():
        model_data.append({
            'Model': model.upper(),
            'Confidence': pred * 100
        })
    
    df = pd.DataFrame(model_data)
    
    fig = px.bar(
        df,
        x='Model',
        y='Confidence',
        title='Model Confidence Levels',
        color='Confidence',
        color_continuous_scale='RdYlBu'
    )
    
    fig.update_layout(
        yaxis_title='Confidence (%)',
        xaxis_title='Model'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model agreement analysis
    st.markdown("#### Model Agreement Analysis")
    agreement_score = calculate_model_agreement(prediction['model_predictions'])
    
    agreement_color = (
        'green' if agreement_score > 0.8 else
        'orange' if agreement_score > 0.6 else
        'red'
    )
    
    st.markdown(
        f"Model Agreement Score: "
        f"<span style='color:{agreement_color}'>{agreement_score:.2%}</span>",
        unsafe_allow_html=True
    )

def display_context_factors(prediction: Dict):
    """Display context factors affecting the prediction."""
    st.markdown("### 📊 Context Factors")
    
    factors = prediction['context_factors']
    
    # Create gauge charts for each factor
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_gauge_chart(
            "Injury Impact",
            factors['injury_impact'],
            "Impact of injuries on prediction"
        )
    
    with col2:
        create_gauge_chart(
            "Conference Factor",
            factors['conference_factor'],
            "Impact of conference strength"
        )
    
    with col3:
        create_gauge_chart(
            "Division Factor",
            factors['division_factor'],
            "Impact of division rivalry"
        )
    
    # Display factor explanations
    st.markdown("#### Factor Explanations")
    st.markdown("""
    - **Injury Impact**: Shows how team injuries affect the prediction
    - **Conference Factor**: Reflects the relative strength of conferences
    - **Division Factor**: Accounts for division rivalry effects
    """)

def create_gauge_chart(title: str, value: float, description: str):
    """Create a gauge chart for factor visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=abs(value) * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ]
        }
    ))
    
    fig.update_layout(height=200)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(description)

def calculate_model_agreement(model_predictions: Dict) -> float:
    """Calculate agreement score between models."""
    predictions = list(model_predictions.values())
    mean_pred = sum(predictions) / len(predictions)
    max_deviation = max(abs(p - mean_pred) for p in predictions)
    return 1 - max_deviation

# 10. Update Main App
def display_predictions(predictions, key=None):
    """Display predictions with enhanced statistics."""
    # Filter out invalid predictions
    valid_predictions = []
    for p in predictions:
        if validate_prediction_structure(p):
            valid_predictions.append(p)
        else:
            logging.warning(f"Invalid prediction structure for game {p.get('game_info', {}).get('id', 'unknown')}")
    
    live_games = [p for p in valid_predictions if p.get('is_live', False)]
    scheduled_games = [p for p in valid_predictions if not p.get('is_live', False)]
    
    # Display metrics using custom containers
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric("Live Games", len(live_games)), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric("Scheduled Games", len(scheduled_games)), unsafe_allow_html=True)
    with col3:
        high_confidence = sum(1 for p in scheduled_games 
                            if p['prediction'].get('confidence_level') == 'High')
        st.markdown(create_metric("High Confidence", high_confidence), unsafe_allow_html=True)
    with col4:
        last_update = datetime.fromtimestamp(st.session_state.last_prediction_time).strftime("%H:%M:%S")
        st.markdown(create_metric("Last Update", last_update), unsafe_allow_html=True)
    
    # Display games
    if live_games:
        st.markdown("## 🔴 Live Games")
        for i, game in enumerate(live_games):
            display_live_game_card(game, key_prefix=f"live_{key}_{i}")
    
    if scheduled_games:
        st.markdown("## 📅 Scheduled Games")
        for i, game in enumerate(scheduled_games):
            display_scheduled_game_card(game, key_prefix=f"scheduled_{key}_{i}")

# 6. Entry point
if __name__ == "__main__":
    initialize_session_state()
    main()




