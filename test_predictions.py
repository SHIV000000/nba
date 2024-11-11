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

def run_continuous_predictions():
    api_key = '89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c'
    api_client = EnhancedNBAApiClient(api_key)
    base_predictor = NBAPredictor('saved_models')
    live_predictor = LiveGamePredictor(base_predictor)
    
    update_interval = 30
    
    while True:
        try:
            live_games = api_client.get_live_games()
            
            if not live_games:
                logging.info("No live games found. Checking today's schedule...")
                today_games = get_todays_schedule(api_client)
                
                if today_games:
                    logging.info(f"Found {len(today_games)} scheduled games for today")
                    process_scheduled_games(today_games, api_client, live_predictor)
                else:
                    logging.info("No games scheduled for today")
                
                sleep(update_interval)
                continue
                
            for game in live_games:
                try:
                    game_info = prepare_game_info(game, api_client)
                    prediction = live_predictor.predict_live_game(game_info)
                    
                    save_prediction(game_info, prediction)
                    log_prediction(game_info, prediction)
                    
                except Exception as e:
                    logging.error(f"Error processing game {game.get('id')}: {str(e)}")
                    continue
                    
            sleep(update_interval)
            
        except KeyboardInterrupt:
            logging.info("Stopping prediction service...")
            break
        except Exception as e:
            logging.error(f"Error in prediction loop: {str(e)}")
            sleep(update_interval)

def prepare_game_info(game: Dict, api_client: EnhancedNBAApiClient) -> Dict:
    """Prepare comprehensive game information."""
    try:
        home_team = game['teams']['home']
        away_team = game['teams']['visitors']
        
        game_info = {
            'gameId': game['id'],
            'home_team': home_team['name'],
            'away_team': away_team['name'],
            'current_period': game['periods']['current'],
            'clock': game['status']['clock'],
            'home_score': int(game['scores']['home']['points']),
            'away_score': int(game['scores']['visitors']['points']),
            'scores': {
                'home': {'linescore': game['scores']['home']['linescore']},
                'away': {'linescore': game['scores']['visitors']['linescore']}
            }
        }
        
        # Get team stats
        home_stats = api_client.get_team_stats(home_team['id'])
        away_stats = api_client.get_team_stats(away_team['id'])
        
        # Try alternative stats if primary fails
        if not home_stats:
            home_stats = api_client.get_team_stats_alternative(home_team['id'])
        if not away_stats:
            away_stats = api_client.get_team_stats_alternative(away_team['id'])
            
        game_info['home_stats'] = home_stats
        game_info['away_stats'] = away_stats
        
        return game_info
        
    except Exception as e:
        logging.error(f"Error preparing game info: {str(e)}")
        raise

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

def save_prediction(game_info: Dict, prediction: Dict):
    """Save prediction with timestamp."""
    try:
        timestamp = datetime.now()
        
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
                'factors': prediction['factors'],
                'models': prediction['model_predictions']
            }
        }
        
        # Create predictions directory if it doesn't exist
        os.makedirs('predictions', exist_ok=True)
        
        # Save to file with timestamp
        filename = f'predictions/game_{game_info["gameId"]}_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
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
    """Process scheduled games and make predictions."""
    try:
        if not games:
            logging.info("No upcoming games to process")
            return
            
        logging.info(f"Processing {len(games)} scheduled games")
        
        for game in games:
            try:
                logging.info(f"Processing game: {game['teams']['home']['name']} vs {game['teams']['visitors']['name']}")
                game_info = prepare_scheduled_game_info(game, api_client)
                prediction = predictor.predict_live_game(game_info)
                
                # Save prediction with scheduled game flag
                save_scheduled_prediction(game_info, prediction)
                log_scheduled_prediction(game_info, prediction)
                
            except Exception as e:
                logging.error(f"Error processing scheduled game {game.get('id')}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error processing scheduled games: {str(e)}")

def prepare_scheduled_game_info(game: Dict, api_client: EnhancedNBAApiClient) -> Dict:
    """Prepare game information for scheduled games."""
    try:
        home_team = game['teams']['home']
        away_team = game['teams']['visitors']
        
        game_info = {
            'gameId': game['id'],
            'home_team': home_team['name'],
            'away_team': away_team['name'],
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
        
        # Get team stats
        home_stats = api_client.get_team_stats(home_team['id'])
        away_stats = api_client.get_team_stats(away_team['id'])
        
        if not home_stats:
            home_stats = api_client.get_team_stats_alternative(home_team['id'])
        if not away_stats:
            away_stats = api_client.get_team_stats_alternative(away_team['id'])
            
        game_info['home_stats'] = home_stats
        game_info['away_stats'] = away_stats
        
        return game_info
        
    except Exception as e:
        logging.error(f"Error preparing scheduled game info: {str(e)}")
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
        
        # Format model predictions
        formatted_predictions = {}
        for model, pred in prediction['model_predictions'].items():
            team_pred = game_info['home_team'] if pred > 0.5 else game_info['away_team']
            prob = pred if pred > 0.5 else (1 - pred)
            formatted_predictions[model] = {
                'predicted_winner': team_pred,
                'win_probability': float(prob)
            }
        
        result = {
            'timestamp': timestamp.isoformat(),
            'game_info': {
                'id': game_info['gameId'],
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'scheduled_start': game_info['scheduled_start'],
                'status': 'Scheduled'
            },
            'prediction': {
                'predicted_winner': predicted_winner,
                'win_probability': float(win_probability),
                'detailed_probabilities': {
                    'home_team': float(home_win_prob),
                    'away_team': float(away_win_prob)
                },
                'model_predictions': formatted_predictions,
                'confidence_level': 'High' if abs(home_win_prob - 0.5) > 0.15 else 'Medium' if abs(home_win_prob - 0.5) > 0.08 else 'Low'
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

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("live_predictions.log"),
            logging.StreamHandler()
        ]
    )
    
    api_client = EnhancedNBAApiClient('89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c')
    
    # Check schedule for next few days first
    check_next_days_schedule(api_client)
    
    # Run the main prediction service
    run_continuous_predictions()



