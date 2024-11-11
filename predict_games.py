# predict_games.py

from api_client import NBAApiClient
from prediction_service import NBAPredictor
import logging
import json
from datetime import datetime
import time
import os

def setup_logging():
    """Setup logging configuration."""
    log_dir = 'logs/predictions'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/predictions_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

def save_prediction(game_info: dict, prediction: float, model_predictions: dict):
    """Save prediction results to JSON file."""
    predictions_dir = 'predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{predictions_dir}/prediction_{game_info["gameId"]}_{timestamp}.json'
    
    result = {
        'timestamp': timestamp,
        'game_info': game_info,
        'ensemble_prediction': float(prediction),
        'model_predictions': {k: float(v) for k, v in model_predictions.items()},
        'home_win_probability': float(prediction),
        'away_win_probability': float(1 - prediction)
    }
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)
    
    return result

def main():
    setup_logging()
    logging.info("Starting NBA game prediction service...")

    # Initialize API client and predictor
    api_key = os.getenv('RAPID_API_KEY')  # Store your API key in environment variables
    if not api_key:
        raise ValueError("RAPID_API_KEY environment variable not set")

    api_client = NBAApiClient(api_key)
    predictor = NBAPredictor("saved_models")

    while True:
        try:
            # Get live games
            live_games = api_client.get_live_games()
            logging.info(f"Found {len(live_games)} live games")
            
            for game in live_games:
                try:
                    # Extract game information
                    game_info = {
                        'gameId': game['gameId'],
                        'home_team': game['hTeam']['fullName'],
                        'away_team': game['vTeam']['fullName'],
                        'current_period': game['currentPeriod'],
                        'clock': game['clock'],
                        'home_score': game['hTeam']['score']['points'],
                        'away_score': game['vTeam']['score']['points']
                    }

                    # Get team IDs
                    home_team_id = game['hTeam']['teamId']
                    away_team_id = game['vTeam']['teamId']

                    # Get team statistics
                    home_stats = api_client.get_team_stats(home_team_id)
                    away_stats = api_client.get_team_stats(away_team_id)

                    # Get current game statistics
                    game_stats = api_client.get_game_statistics(game['gameId'])
                    
                    # Combine season and game statistics
                    home_combined_stats = combine_stats(home_stats, game_stats, home_team_id)
                    away_combined_stats = combine_stats(away_stats, game_stats, away_team_id)

                    # Make prediction
                    prediction, model_predictions = predictor.predict_game(
                        home_combined_stats, 
                        away_combined_stats
                    )

                    # Save and log prediction
                    result = save_prediction(game_info, prediction, model_predictions)
                    
                    logging.info(f"""
                    Game: {game_info['home_team']} vs {game_info['away_team']}
                    Current Score: {game_info['home_score']} - {game_info['away_score']}
                    Prediction: {result['home_win_probability']:.2%} chance of home team win
                    """)

                except Exception as e:
                    logging.error(f"Error processing game {game.get('gameId', 'unknown')}: {str(e)}")
                    continue

            # Wait before next update
            time.sleep(60)  # Update every minute

        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            time.sleep(30)  # Wait before retrying

def combine_stats(season_stats: dict, game_stats: dict, team_id: str) -> dict:
    """Combine season statistics with current game statistics."""
    try:
        # Find current game stats for the team
        current_game_stats = next(
            (stats for stats in game_stats if stats['teamId'] == team_id),
            {}
        )

        # Calculate weighted average of season and current game stats
        combined_stats = {}
        stat_keys = [
            'points', 'fgp', 'tpp', 'ftp', 'totReb', 'assists',
            'steals', 'blocks', 'turnovers', 'fouls'
        ]

        for key in stat_keys:
            season_value = float(season_stats.get(key, 0))
            game_value = float(current_game_stats.get(key, 0))
            
            # Weight current game stats more heavily
            combined_stats[key] = (0.7 * season_value + 0.3 * game_value)

        # Add additional advanced metrics
        combined_stats['offensive_efficiency'] = calculate_offensive_efficiency(combined_stats)
        combined_stats['defensive_efficiency'] = calculate_defensive_efficiency(combined_stats)
        
        return combined_stats

    except Exception as e:
        logging.error(f"Error combining stats: {str(e)}")
        return season_stats

def calculate_offensive_efficiency(stats: dict) -> float:
    """Calculate offensive efficiency rating."""
    try:
        points = stats.get('points', 0)
        possessions = (
            stats.get('fga', 0) - stats.get('offReb', 0) + 
            stats.get('turnovers', 0) + (0.4 * stats.get('fta', 0))
        )
        return (points * 100) / possessions if possessions > 0 else 0
    except:
        return 0

def calculate_defensive_efficiency(stats: dict) -> float:
    """Calculate defensive efficiency rating."""
    try:
        points_allowed = stats.get('points_allowed', 0)
        possessions = (
            stats.get('fga', 0) - stats.get('offReb', 0) + 
            stats.get('turnovers', 0) + (0.4 * stats.get('fta', 0))
        )
        return (points_allowed * 100) / possessions if possessions > 0 else 0
    except:
        return 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Prediction service stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
