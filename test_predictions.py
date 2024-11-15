# test_predictions.py

import logging
import json
from datetime import datetime
import os
from time import sleep
from api_client import EnhancedNBAApiClient
from prediction_service import NBAPredictor
from typing import Dict, Any, List
from datetime import datetime, timedelta
from typing import Optional
import time
import atexit
import logging
from datetime import datetime, timedelta

import atexit
import shutil
import threading
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(process)d - %(threadName)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("live_predictions.log"),
        logging.StreamHandler()
    ]
)

class LiveGamePredictor:
    def __init__(self, base_predictor: NBAPredictor):
        self.base_predictor = base_predictor
        self.game_cache = {}
        self.update_interval = 30  # seconds

    def predict_live_game(self, game_info: Dict) -> Dict[str, Any]:
        """Make and update predictions for a live game."""
        game_id = game_info['gameId']
        current_time = datetime.now()

        if self._should_update_prediction(game_id, current_time):
            prediction = self._make_live_prediction(game_info)
            self.game_cache[game_id] = {
                'last_update': current_time,
                'prediction': prediction
            }
            return prediction
        
        return self.game_cache[game_id]['prediction']

    def _should_update_prediction(self, game_id: str, current_time: datetime) -> bool:
        if game_id not in self.game_cache:
            return True

        last_update = self.game_cache[game_id]['last_update']
        time_diff = (current_time - last_update).total_seconds()
        return time_diff >= self.update_interval

    def _make_live_prediction(self, game_info: Dict) -> Dict[str, Any]:
        base_prediction, model_predictions = self.base_predictor.predict_game(
            game_info['home_stats'],
            game_info['away_stats']
        )

        momentum_factor = self._calculate_momentum(game_info)
        performance_factor = self._calculate_performance_factor(game_info)
        time_pressure = self._calculate_time_pressure(game_info)

        adjusted_prediction = self._adjust_prediction(
            base_prediction,
            momentum_factor,
            performance_factor,
            time_pressure,
            game_info
        )

        return {
            'base_prediction': base_prediction,
            'adjusted_prediction': adjusted_prediction,
            'model_predictions': model_predictions,
            'factors': {
                'momentum': momentum_factor,
                'performance': performance_factor,
                'time_pressure': time_pressure
            },
            'game_state': {
                'period': game_info['current_period'],
                'time_remaining': self._parse_game_clock(game_info['clock']),
                'score_difference': game_info['home_score'] - game_info['away_score']
            }
        }

    def _calculate_momentum(self, game_info: Dict) -> float:
        try:
            home_scores = [int(score) for score in game_info['scores']['home']['linescore']]
            away_scores = [int(score) for score in game_info['scores']['away']['linescore']]
            
            recent_home = sum(home_scores[-2:])
            recent_away = sum(away_scores[-2:])
            
            momentum = (recent_home - recent_away) / max(recent_home + recent_away, 1)
            return max(min(momentum, 1.0), -1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating momentum: {str(e)}")
            return 0.0

    def _calculate_performance_factor(self, game_info: Dict) -> float:
        try:
            home_stats = game_info['home_stats']['statistics'][0]
            away_stats = game_info['away_stats']['statistics'][0]
            
            home_ppg = float(home_stats.get('points', 0))
            away_ppg = float(away_stats.get('points', 0))
            
            current_home_pace = game_info['home_score'] / max(game_info['current_period'], 1)
            current_away_pace = game_info['away_score'] / max(game_info['current_period'], 1)
            
            home_performance = current_home_pace / home_ppg if home_ppg > 0 else 1.0
            away_performance = current_away_pace / away_ppg if away_ppg > 0 else 1.0
            
            return home_performance - away_performance
            
        except Exception as e:
            logging.warning(f"Error calculating performance factor: {str(e)}")
            return 0.0

    def _calculate_time_pressure(self, game_info: Dict) -> float:
        try:
            total_minutes = 48.0
            current_minute = (game_info['current_period'] - 1) * 12
            
            if game_info['clock']:
                minutes, seconds = map(float, game_info['clock'].split(':'))
                current_minute += (12 - minutes - seconds/60)
            
            return min(current_minute / total_minutes, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating time pressure: {str(e)}")
            return 0.0

    def _parse_game_clock(self, clock_str: str) -> float:
        try:
            if not clock_str:
                return 12.0
            minutes, seconds = map(float, clock_str.split(':'))
            return minutes + (seconds / 60)
        except Exception as e:
            logging.warning(f"Error parsing game clock: {str(e)}")
            return 12.0

    def _adjust_prediction(
        self,
        base_pred: float,
        momentum: float,
        performance: float,
        time_pressure: float,
        game_info: Dict
    ) -> float:
        try:
            momentum_weight = 0.2
            performance_weight = 0.3
            score_weight = 0.5
            
            score_diff = game_info['home_score'] - game_info['away_score']
            max_diff = 20.0
            score_factor = max(min(score_diff / max_diff, 1.0), -1.0)
            
            adjustment = (
                momentum * momentum_weight +
                performance * performance_weight +
                score_factor * score_weight
            ) * time_pressure
            
            adjusted_pred = base_pred + (adjustment * (1 - base_pred))
            return max(min(adjusted_pred, 1.0), 0.0)
            
        except Exception as e:
            logging.warning(f"Error adjusting prediction: {str(e)}")
            return base_pred

def run_continuous_predictions(timeout_minutes=3):
    """Run predictions with timeout"""
    api_key = '89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c'
    api_client = EnhancedNBAApiClient(api_key)
    base_predictor = NBAPredictor('saved_models')
    live_predictor = LiveGamePredictor(base_predictor)
    
    start_time = time.time()
    update_interval = 30
    
    try:
        while True:
            # Check if timeout has been reached
            if time.time() - start_time > (timeout_minutes * 60):
                logging.info(f"Prediction service timeout reached ({timeout_minutes} minutes)")
                break
                
            live_games = api_client.get_live_games()
            predictions_made = False
            
            if live_games:
                for game in live_games:
                    try:
                        game_info = prepare_game_info(game, api_client)
                        prediction = live_predictor.predict_live_game(game_info)
                        
                        # Save to live predictions directory
                        save_prediction(game_info, prediction, is_live=True)
                        log_prediction(game_info, prediction)
                        predictions_made = True
                    except Exception as e:
                        logging.error(f"Error processing game {game.get('id')}: {str(e)}")
                        continue
            else:
                today_games = get_todays_schedule(api_client)
                if today_games:
                    process_scheduled_games(today_games, api_client, live_predictor)
                    predictions_made = True
                else:
                    logging.info("No games found")
            
            if predictions_made:
                logging.info("Predictions completed successfully")
                break
            
            # Sleep for shorter interval to allow for more responsive timeout
            time.sleep(min(update_interval, 10))
            
    except Exception as e:
        logging.error(f"Error in prediction service: {str(e)}")
        return False
    finally:
        logging.info("Prediction service stopped")
        return True

def prepare_game_info(game: Dict, api_client: EnhancedNBAApiClient) -> Dict:
    """Prepare comprehensive game information."""
    try:
        teams = game.get('teams', {})
        home_team = teams.get('home', {})
        away_team = teams.get('away', {})
        
        # Get team information
        home_team_info = api_client.get_team_info(home_team.get('id'))
        away_team_info = api_client.get_team_info(away_team.get('id'))
        
        # Get injury information
        home_injuries = api_client.get_team_injuries(home_team.get('id'))
        away_injuries = api_client.get_team_injuries(away_team.get('id'))
        
        game_info = {
            'gameId': game.get('id'),
            'home_team': {
                'name': home_team.get('name'),
                'info': home_team_info,
                'injuries': home_injuries
            },
            'away_team': {
                'name': away_team.get('name'),
                'info': away_team_info,
                'injuries': away_injuries
            },
            'current_period': game.get('periods', {}).get('current', 1),
            'clock': game.get('status', {}).get('clock', '12:00'),
            'home_score': int(game.get('scores', {}).get('home', {}).get('points', 0)),
            'away_score': int(game.get('scores', {}).get('away', {}).get('points', 0))
        }
        
        # Get and process team stats
        home_stats = api_client.get_team_stats(home_team.get('id'))
        away_stats = api_client.get_team_stats(away_team.get('id'))
        
        # Adjust stats based on injuries
        home_stats = adjust_stats_for_injuries(home_stats, home_injuries)
        away_stats = adjust_stats_for_injuries(away_stats, away_injuries)
        
        game_info['home_stats'] = home_stats
        game_info['away_stats'] = away_stats
        
        return game_info
        
    except Exception as e:
        logging.error(f"Error preparing game info: {str(e)}")
        raise

def adjust_stats_for_injuries(stats: Dict, injuries: List[Dict]) -> Dict:
    """Adjust team statistics based on injured players."""
    try:
        if not injuries or not stats.get('statistics'):
            return stats
            
        # Calculate impact factor based on number and status of injuries
        impact_factor = sum(
            0.1 if injury['status'] == 'Out' else 0.05
            for injury in injuries
        )
        
        # Adjust statistics
        adjusted_stats = stats.copy()
        stat_keys = ['points', 'assists', 'rebounds', 'steals', 'blocks']
        
        for key in stat_keys:
            if key in adjusted_stats['statistics'][0]:
                adjusted_stats['statistics'][0][key] *= (1 - impact_factor)
                
        return adjusted_stats
        
    except Exception as e:
        logging.error(f"Error adjusting stats for injuries: {str(e)}")
        return stats

def log_prediction(game_info: Dict, prediction: Dict):
    """Log detailed prediction information."""
    logging.info(f"""
    ============ Game Update ============
    {game_info['home_team']} vs {game_info['away_team']}
    Period: {game_info['current_period']} | Time: {game_info['clock']}
    Score: {game_info['home_score']} - {game_info['away_score']}
    
    Predictions:
    - Base: {prediction['base_prediction']:.2%}
    - Adjusted: {prediction['adjusted_prediction']:.2%}
    
    Adjustment Factors:
    - Momentum: {prediction['factors']['momentum']:.3f}
    - Performance: {prediction['factors']['performance']:.3f}
    - Time Pressure: {prediction['factors']['time_pressure']:.3f}
    
    Model Predictions:
    {json.dumps(prediction['model_predictions'], indent=2)}
    
    Game State:
    - Time Remaining: {prediction['game_state']['time_remaining']:.1f} minutes
    - Score Difference: {prediction['game_state']['score_difference']} points
    =====================================
    """)

def save_prediction(game_info: Dict, prediction: Dict, is_live: bool = True):
    """Save prediction with timestamp."""
    try:
        timestamp = datetime.now()
        
        # Calculate predicted winner
        home_win_prob = prediction['adjusted_prediction']
        predicted_winner = game_info['home_team'] if home_win_prob > 0.5 else game_info['away_team']
        win_probability = home_win_prob if home_win_prob > 0.5 else (1 - home_win_prob)
        
        result = {
            'timestamp': timestamp.isoformat(),
            'game_info': {
                'id': game_info['gameId'],
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'period': game_info['current_period'],
                'clock': game_info['clock'],
                'score': {
                    'home': game_info['home_score'],
                    'away': game_info['away_score']
                }
            },
            'prediction': {
                'base': float(prediction['base_prediction']),
                'adjusted': float(prediction['adjusted_prediction']),
                'predicted_winner': predicted_winner,
                'win_probability': float(win_probability),
                'factors': prediction['factors'],
                'models': prediction['model_predictions']
            },
            'is_live': is_live
        }
        
        directory = 'predictions/live' if is_live else 'predictions/scheduled'
        os.makedirs(directory, exist_ok=True)
        
        filename = f'{directory}/game_{game_info["gameId"]}_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
            
        logging.debug(f"Saved prediction to {filename}")
        
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}")

def debug_game_status(game: Dict) -> str:
    """Helper function to debug game status"""
    return (f"Game ID: {game.get('id')} - "
            f"Status: {game.get('status', {}).get('long')} - "
            f"Clock: {game.get('status', {}).get('clock')} - "
            f"Period: {game.get('periods', {}).get('current')} - "
            f"{game.get('teams', {}).get('home', {}).get('name')} "
            f"vs {game.get('teams', {}).get('visitors', {}).get('name')}")

def get_todays_schedule(api_client: EnhancedNBAApiClient) -> List[Dict]:
    """Get today's game schedule."""
    try:
        endpoint = f"{api_client.base_url}/games"
        
        # Get today's date and next few days
        today = datetime.now()
        dates_to_try = [
            (today + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(3)  # Try today and next 2 days
        ]
        
        all_games = []
        season = "2024"  # Fixed for 2024-2025 season
        
        for date in dates_to_try:
            params = {
                'date': date,
                'season': season,
                'league': 'standard'
            }
            
            logging.debug(f"Fetching schedule for date: {date}, season: {season}")
            
            response = api_client._make_request('GET', endpoint, params)
            games = response.get('response', [])
            
            if games:
                all_games.extend(games)
                
        logging.debug(f"Found {len(all_games)} total games")
        
        # Filter and process games
        processed_games = []
        for game in all_games:
            game_date_str = game.get('date', {}).get('start')
            if not game_date_str:
                continue
                
            try:
                game_date = datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                game_date = game_date.replace(tzinfo=None)  # Remove timezone info for comparison
                
                # Only include games that are in the future
                if game_date > datetime.now():
                    processed_game = {
                        'id': game.get('id'),
                        'teams': {
                            'home': {
                                'id': game.get('teams', {}).get('home', {}).get('id'),
                                'name': game.get('teams', {}).get('home', {}).get('name')
                            },
                            'visitors': {
                                'id': game.get('teams', {}).get('visitors', {}).get('id'),
                                'name': game.get('teams', {}).get('visitors', {}).get('name')
                            }
                        },
                        'date': {
                            'start': game_date_str,
                        },
                        'status': {
                            'long': game.get('status', {}).get('long'),
                            'short': game.get('status', {}).get('short')
                        },
                        'arena': game.get('arena', {})
                    }
                    processed_games.append(processed_game)
                    
                    logging.info(f"Found upcoming game: {processed_game['teams']['home']['name']} vs "
                               f"{processed_game['teams']['visitors']['name']} at {game_date_str}")
            except Exception as e:
                logging.error(f"Error processing game date {game_date_str}: {str(e)}")
                continue
        
        if processed_games:
            logging.info(f"Found {len(processed_games)} upcoming games")
        else:
            logging.info("No upcoming games found after filtering")
            
        return processed_games
        
    except Exception as e:
        logging.error(f"Error fetching today's schedule: {str(e)}")
        logging.exception("Full traceback:")
        return []

def parse_game_time(time_str: str) -> Optional[datetime]:
    """Parse game time string to datetime object."""
    try:
        if not time_str:
            return None
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception as e:
        logging.error(f"Error parsing game time {time_str}: {str(e)}")
        return None

def process_scheduled_games(games: List[Dict], api_client: EnhancedNBAApiClient, predictor: LiveGamePredictor):
    """Process scheduled games with prediction display."""
    for game in games:
        try:
            # Extract team IDs with proper path navigation
            teams = game.get('teams', {})
            home_team = teams.get('home', {}) or teams.get('homeTeam', {})
            away_team = teams.get('visitors', {}) or teams.get('away', {}) or teams.get('awayTeam', {})
            
            home_id = str(home_team.get('id', ''))
            away_id = str(away_team.get('id', ''))
            
            # Log team information for debugging
            logging.debug(f"Processing game {game.get('id')}")
            logging.debug(f"Home team: {home_team.get('name')} (ID: {home_id})")
            logging.debug(f"Away team: {away_team.get('name')} (ID: {away_id})")
            
            if not home_id or not away_id:
                logging.warning(f"Missing team ID for game {game.get('id')}")
                logging.debug(f"Raw game data: {json.dumps(game, indent=2)}")
                continue
                
            # Get team stats with retry logic
            home_stats = get_team_stats_with_retry(api_client, home_id)
            away_stats = get_team_stats_with_retry(api_client, away_id)
            
            # Prepare game info
            game_info = {
                'gameId': game.get('id'),
                'home_team': home_team.get('name'),
                'away_team': away_team.get('name'),
                'home_stats': home_stats,
                'away_stats': away_stats,
                'scheduled_start': game.get('date', {}).get('start'),
                'current_period': 0,
                'clock': '12:00',
                'home_score': 0,
                'away_score': 0,
                'scores': {
                    'home': {'linescore': [0]},
                    'away': {'linescore': [0]}
                }
            }
            
            # Make prediction
            prediction = predictor.predict_live_game(game_info)
            
            # Display prediction summary
            display_prediction_summary(game_info, prediction)
            
            # Save prediction
            save_scheduled_prediction(game_info, prediction)
            
        except Exception as e:
            logging.error(f"Error processing game {game.get('id')}: {str(e)}")
            continue

def get_team_stats_with_retry(api_client: EnhancedNBAApiClient, team_id: str, max_retries: int = 3) -> Dict:
    """Get team stats with retry logic."""
    for attempt in range(max_retries):
        try:
            stats = api_client.get_team_stats(team_id)
            if stats and stats.get('statistics'):
                return stats
            
            # Try alternative endpoint if primary fails
            stats = api_client.get_team_stats_alternative(team_id)
            if stats and stats.get('statistics'):
                return stats
                
            time.sleep(1)  # Short delay between retries
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for team ID {team_id}: {str(e)}")
            
    return {'statistics': [{}]}  # Return empty stats if all attempts fail

def prepare_scheduled_game_info(game: Dict, api_client: EnhancedNBAApiClient) -> Dict:
    """Prepare game information for scheduled games."""
    try:
        teams = game.get('teams', {})
        home_team = teams.get('home', {})
        away_team = teams.get('away', {})  # Changed from 'visitors' to 'away'
        
        game_info = {
            'gameId': game.get('id'),
            'home_team': home_team.get('name'),
            'away_team': away_team.get('name'),
            'current_period': 0,
            'clock': '12:00',
            'home_score': 0,
            'away_score': 0,
            'scores': {
                'home': {'linescore': [0]},
                'away': {'linescore': [0]}
            },
            'scheduled_start': game.get('date', {}).get('start'),
            'status': 'Scheduled'
        }
        
        # Get team stats with error handling
        try:
            home_stats = api_client.get_team_stats(home_team.get('id'))
        except Exception as e:
            logging.warning(f"Error getting home team stats: {str(e)}")
            home_stats = {'statistics': [{}]}
            
        try:
            away_stats = api_client.get_team_stats(away_team.get('id'))
        except Exception as e:
            logging.warning(f"Error getting away team stats: {str(e)}")
            away_stats = {'statistics': [{}]}
        
        if not home_stats or not home_stats.get('statistics'):
            try:
                home_stats = api_client.get_team_stats_alternative(home_team.get('id'))
            except Exception as e:
                logging.warning(f"Error getting alternative home team stats: {str(e)}")
                home_stats = {'statistics': [{}]}
                
        if not away_stats or not away_stats.get('statistics'):
            try:
                away_stats = api_client.get_team_stats_alternative(away_team.get('id'))
            except Exception as e:
                logging.warning(f"Error getting alternative away team stats: {str(e)}")
                away_stats = {'statistics': [{}]}
            
        game_info['home_stats'] = home_stats
        game_info['away_stats'] = away_stats
        
        return game_info
        
    except Exception as e:
        logging.error(f"Error preparing scheduled game info: {str(e)}")
        logging.debug(f"Raw game data: {json.dumps(game, indent=2)}")
        raise

def save_scheduled_prediction(game_info: Dict, prediction: Dict):
    """Save detailed prediction for scheduled game."""
    try:
        timestamp = datetime.now()
        
        # Calculate win probabilities and predictions
        home_win_prob = prediction['base_prediction']
        away_win_prob = 1 - home_win_prob
        predicted_winner = game_info['home_team'] if home_win_prob > 0.5 else game_info['away_team']
        win_probability = home_win_prob if home_win_prob > 0.5 else away_win_prob
        
        # Calculate score predictions based on team stats
        home_stats = game_info['home_stats']['statistics'][0]
        away_stats = game_info['away_stats']['statistics'][0]
        
        # Get points per game and calculate ranges
        home_ppg = float(home_stats.get('points', 100))  # Default to 100 if not found
        away_ppg = float(away_stats.get('points', 100))
        
        # Calculate score ranges with margin based on win probability
        margin_factor = abs(home_win_prob - 0.5) * 10  # Max 5-point margin
        
        # Adjust ranges based on predicted winner
        if home_win_prob > 0.5:
            home_margin = margin_factor
            away_margin = -margin_factor
        else:
            home_margin = -margin_factor
            away_margin = margin_factor
            
        score_prediction = {
            'home_low': max(int(home_ppg + home_margin - 5), 0),
            'home_high': max(int(home_ppg + home_margin + 5), 0),
            'away_low': max(int(away_ppg + away_margin - 5), 0),
            'away_high': max(int(away_ppg + away_margin + 5), 0)
        }
        
        result = {
            'timestamp': timestamp.isoformat(),
            'game_info': {
                'id': game_info['gameId'],
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'scheduled_start': game_info['scheduled_start']
            },
            'prediction': {
                'predicted_winner': predicted_winner,
                'win_probability': float(win_probability),
                'score_prediction': score_prediction,  # Add score prediction
                'key_factors': {
                    'home_team_stats': home_stats,
                    'away_team_stats': away_stats
                },
                'model_predictions': prediction['model_predictions'],
                'confidence_level': 'High' if win_probability > 0.65 else 'Medium' if win_probability > 0.55 else 'Low'
            }
        }
        
        os.makedirs('predictions/scheduled', exist_ok=True)
        filename = f'predictions/scheduled/game_{game_info["gameId"]}_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
            
        logging.debug(f"Saved detailed scheduled game prediction to {filename}")
        
    except Exception as e:
        logging.error(f"Error saving scheduled prediction: {str(e)}")

def log_scheduled_prediction(game_info: Dict, prediction: Dict):
    """Log prediction for scheduled game with clear winner prediction."""
    # Calculate win probabilities
    home_win_prob = prediction['base_prediction']
    away_win_prob = 1 - home_win_prob
    
    # Determine predicted winner
    predicted_winner = game_info['home_team'] if home_win_prob > 0.5 else game_info['away_team']
    win_probability = home_win_prob if home_win_prob > 0.5 else away_win_prob
    
    # Calculate model consensus
    model_predictions = prediction['model_predictions']
    home_votes = sum(1 for pred in model_predictions.values() if pred > 0.5)
    away_votes = len(model_predictions) - home_votes
    
    # Format model predictions with clear labels
    formatted_predictions = {}
    for model, pred in model_predictions.items():
        team_pred = game_info['home_team'] if pred > 0.5 else game_info['away_team']
        prob = pred if pred > 0.5 else (1 - pred)
        formatted_predictions[model] = {
            'predicted_winner': team_pred,
            'win_probability': f"{prob:.2%}"
        }
    
    logging.info(f"""
    ========= Game Prediction Analysis =========
    Match: {game_info['home_team']} (Home) vs {game_info['away_team']} (Away)
    Scheduled Start: {game_info['scheduled_start']}
    
    PREDICTION SUMMARY:
    Predicted Winner: {predicted_winner}
    Win Probability: {win_probability:.2%}
    
    Detailed Probabilities:
    - {game_info['home_team']}: {home_win_prob:.2%}
    - {game_info['away_team']}: {away_win_prob:.2%}
    
    Model Consensus:
    - Models favoring {game_info['home_team']}: {home_votes}
    - Models favoring {game_info['away_team']}: {away_votes}
    
    Individual Model Predictions:
    {json.dumps(formatted_predictions, indent=2)}
    
    Key Factors:
    - Home Court Advantage: Considered in base prediction
    - Historical Performance: Based on current season statistics
    - Team Form: Using recent game statistics
    
    Confidence Level: {'High' if abs(home_win_prob - 0.5) > 0.15 else 'Medium' if abs(home_win_prob - 0.5) > 0.08 else 'Low'}
    Model Agreement: {'Strong' if abs(home_votes - away_votes) >= 3 else 'Moderate' if abs(home_votes - away_votes) >= 2 else 'Split'}
    ==========================================
    """)

def check_schedule_directly(api_client: EnhancedNBAApiClient, date_str: str):
    """Debug helper to check schedule directly."""
    try:
        endpoint = f"{api_client.base_url}/games"
        params = {
            'date': date_str,
            'season': '2024',
            'league': 'standard'
        }
        
        response = api_client._make_request('GET', endpoint, params)
        print(f"\nSchedule check for {date_str}:")
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        print(f"Error checking schedule: {str(e)}")

def check_next_days_schedule(api_client: EnhancedNBAApiClient, days: int = 3):
    """Debug helper to check schedule for next few days."""
    today = datetime.now()
    
    print("\nChecking schedule for next few days:")
    for i in range(days):
        check_date = today + timedelta(days=i)
        date_str = check_date.strftime("%Y-%m-%d")
        
        try:
            endpoint = f"{api_client.base_url}/games"
            params = {
                'date': date_str,
                'season': '2024',
                'league': 'standard'
            }
            
            response = api_client._make_request('GET', endpoint, params)
            games = response.get('response', [])
            
            print(f"\nDate: {date_str}")
            print(f"Found {len(games)} games")
            
            for game in games:
                game_time = game.get('date', {}).get('start')
                home_team = game.get('teams', {}).get('home', {}).get('name')
                away_team = game.get('teams', {}).get('visitors', {}).get('name')
                status = game.get('status', {}).get('long')
                
                print(f"- {away_team} @ {home_team}")
                print(f"  Time: {game_time}")
                print(f"  Status: {status}")
                
        except Exception as e:
            print(f"Error checking {date_str}: {str(e)}")

def display_prediction_summary(game_info: Dict, prediction: Dict):
    """Display a comprehensive game prediction with detailed analysis."""
    home_win_prob = prediction['base_prediction']
    away_win_prob = 1 - home_win_prob
    
    # Calculate predicted score range based on team stats
    home_stats = game_info['home_stats']['statistics'][0]
    away_stats = game_info['away_stats']['statistics'][0]
    
    home_avg_points = float(home_stats.get('points', 0))
    away_avg_points = float(away_stats.get('points', 0))
    
    # Score range calculation (±5 points from average)
    home_score_range = (max(int(home_avg_points - 5), 0), int(home_avg_points + 5))
    away_score_range = (max(int(away_avg_points - 5), 0), int(away_avg_points + 5))

    print(f"\n{'='*50}")
    print(f"GAME PREDICTION ANALYSIS")
    print(f"{'='*50}")
    
    print(f"\n🏀 MATCHUP:")
    print(f"{game_info['home_team']} (Home) vs {game_info['away_team']} (Away)")
    
    print(f"\n📊 WIN PROBABILITY:")
    print(f"{game_info['home_team']}: {home_win_prob:.1%}")
    print(f"{game_info['away_team']}: {away_win_prob:.1%}")
    
    print(f"\n🎯 PREDICTED SCORE RANGE:")
    print(f"{game_info['home_team']}: {home_score_range[0]}-{home_score_range[1]} points")
    print(f"{game_info['away_team']}: {away_score_range[0]}-{away_score_range[1]} points")
    
    predicted_winner = game_info['home_team'] if home_win_prob > 0.5 else game_info['away_team']
    win_prob = max(home_win_prob, away_win_prob)
    
    print(f"\n🏆 PREDICTION SUMMARY:")
    print(f"Predicted Winner: {predicted_winner}")
    print(f"Win Confidence: {win_prob:.1%}")
    
    # Calculate confidence level
    confidence = "High" if win_prob > 0.65 else "Medium" if win_prob > 0.55 else "Low"
    print(f"Confidence Level: {confidence}")
    
    print(f"\n📈 KEY FACTORS:")
    # Team Statistics Comparison
    print("Team Statistics (Season Averages):")
    stats_to_compare = [
        ('Points', 'points'),
        ('FG%', 'fieldGoalsPercentage'),
        ('3P%', 'threePointsPercentage'),
        ('Rebounds', 'reboundsTotal'),
        ('Assists', 'assists')
    ]
    
    for stat_name, stat_key in stats_to_compare:
        home_val = float(home_stats.get(stat_key, 0))
        away_val = float(away_stats.get(stat_key, 0))
        print(f"- {stat_name}: {game_info['home_team']}: {home_val:.1f} | {game_info['away_team']}: {away_val:.1f}")
    
    print(f"\n🤖 MODEL PREDICTIONS:")
    for model, pred in prediction['model_predictions'].items():
        team = game_info['home_team'] if pred > 0.5 else game_info['away_team']
        prob = pred if pred > 0.5 else (1 - pred)
        confidence = "High" if prob > 0.65 else "Medium" if prob > 0.55 else "Low"
        print(f"- {model.upper()}: {team} ({prob:.1%}) - {confidence} Confidence")
    
    print(f"\n{'='*50}")
    print("Note: Predictions are based on historical data and current team performance.")
    print("Actual results may vary due to game-day factors and circumstances.")
    print(f"{'='*50}\n")

def display_team_analysis(game_info: Dict):
    """Display comprehensive team analysis including injuries and stats."""
    try:
        for team_type in ['home_team', 'away_team']:
            team = game_info[team_type]
            print(f"\n{'='*20} {team['name']} Analysis {'='*20}")
            
            # Team Information
            print("\n📋 Team Information:")
            info = team['info']
            print(f"Conference: {info['conference']}")
            print(f"Division: {info['division']}")
            print(f"City: {info['city']}")
            
            # Injury Report
            print("\n🏥 Injury Report:")
            if team['injuries']:
                for injury in team['injuries']:
                    print(f"- {injury['player']}: {injury['status']} - {injury['reason']}")
            else:
                print("No reported injuries")
            
            # Team Statistics
            stats = team['stats']['statistics'][0]
            print("\n📊 Season Statistics:")
            print(f"Points per Game: {stats.get('points', 0):.1f}")
            print(f"FG%: {stats.get('fieldGoalsPercentage', 0):.1f}%")
            print(f"3P%: {stats.get('threePointsPercentage', 0):.1f}%")
            print(f"Rebounds: {stats.get('reboundsTotal', 0):.1f}")
            print(f"Assists: {stats.get('assists', 0):.1f}")
            
    except Exception as e:
        logging.error(f"Error displaying team analysis: {str(e)}")

if __name__ == "__main__":
    api_client = EnhancedNBAApiClient('89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c')
    
    # Check schedule
    check_next_days_schedule(api_client)
    
    # Get upcoming games
    upcoming_games = get_todays_schedule(api_client)
    
    if upcoming_games:
        # Initialize predictor
        base_predictor = NBAPredictor('saved_models')
        live_predictor = LiveGamePredictor(base_predictor)
        
        # Process and display predictions
        process_scheduled_games(upcoming_games, api_client, live_predictor)
    else:
        print("No upcoming games found")






