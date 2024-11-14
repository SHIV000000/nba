# reward_system.py

from typing import Dict, Tuple, List
from collections import defaultdict
import numpy as np
import logging
from datetime import datetime
import json
import os

class PredictionRewardSystem:
    def __init__(self, season_year: str = "2024"):
        self.season_year = season_year
        self.total_coins_goal = 2460
        self.total_boost_goal = 2337  # 95% of total games
        self.current_coins = 0
        self.current_boost_points = 0
        self.prediction_history = []
        self.learning_rate = 0.01
        self.score_margin_threshold = 3
        
        # Create directories for storing results
        self.results_dir = "prediction_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load existing progress if available
        self._load_progress()

    def calculate_reward(self, prediction: Dict, actual_result: Dict) -> Tuple[int, int, Dict]:
        """
        Calculate rewards for prediction accuracy
        Returns: (coins_earned, boost_points, analysis)
        """
        try:
            coins_earned = 0
            boost_points = 0
            
            # Extract predicted and actual scores
            pred_home_score = prediction['game_info']['score']['home']
            pred_away_score = prediction['game_info']['score']['away']
            actual_home_score = actual_result['home_score']
            actual_away_score = actual_result['away_score']
            
            # Determine winners
            predicted_winner = prediction['prediction']['predicted_winner']
            actual_winner = actual_result['winner']
            
            # Calculate win/loss reward
            if predicted_winner == actual_winner:
                coins_earned = 1
                
            # Calculate score prediction accuracy
            home_score_diff = abs(pred_home_score - actual_home_score)
            away_score_diff = abs(pred_away_score - actual_away_score)
            
            # Award boost point if prediction is within margin
            if home_score_diff <= self.score_margin_threshold and away_score_diff <= self.score_margin_threshold:
                boost_points = 1
                
            analysis = {
                'prediction_accuracy': {
                    'winner_correct': predicted_winner == actual_winner,
                    'home_score_diff': home_score_diff,
                    'away_score_diff': away_score_diff,
                    'within_margin': boost_points == 1
                },
                'reward_earned': {
                    'coins': coins_earned,
                    'boost_points': boost_points
                }
            }
            
            return coins_earned, boost_points, analysis
            
        except Exception as e:
            logging.error(f"Error calculating reward: {str(e)}")
            return 0, 0, {'error': str(e)}

    def update_metrics(self, game_id: str, prediction: Dict, actual_result: Dict):
        """Update system metrics with new prediction results"""
        try:
            coins, boost, analysis = self.calculate_reward(prediction, actual_result)
            
            self.current_coins += coins
            self.current_boost_points += boost
            
            result_entry = {
                'timestamp': datetime.now().isoformat(),
                'game_id': game_id,
                'prediction': prediction,
                'actual_result': actual_result,
                'rewards': {
                    'coins_earned': coins,
                    'boost_points': boost
                },
                'analysis': analysis
            }
            
            self.prediction_history.append(result_entry)
            self._save_result(result_entry)
            self._save_progress()
            
            # Adjust learning parameters based on performance
            self._adjust_learning_parameters()
            
        except Exception as e:
            logging.error(f"Error updating metrics: {str(e)}")

    def get_progress_report(self) -> Dict:
        """Get detailed progress report"""
        try:
            games_processed = len(self.prediction_history)
            recent_accuracy = self._calculate_recent_accuracy()
            
            return {
                'coins': {
                    'current': self.current_coins,
                    'goal': self.total_coins_goal,
                    'remaining': self.total_coins_goal - self.current_coins,
                    'progress_percentage': (self.current_coins / self.total_coins_goal) * 100
                },
                'boost_points': {
                    'current': self.current_boost_points,
                    'goal': self.total_boost_goal,
                    'remaining': self.total_boost_goal - self.current_boost_points,
                    'progress_percentage': (self.current_boost_points / self.total_boost_goal) * 100
                },
                'performance_metrics': {
                    'games_processed': games_processed,
                    'recent_accuracy': recent_accuracy,
                    'learning_rate': self.learning_rate
                }
            }
            
        except Exception as e:
            logging.error(f"Error generating progress report: {str(e)}")
            return {}

    def _calculate_recent_accuracy(self, window: int = 10) -> Dict:
        """Calculate recent prediction accuracy"""
        try:
            if not self.prediction_history:
                return {'win_rate': 0.0, 'boost_rate': 0.0}
                
            recent_games = self.prediction_history[-window:]
            win_count = sum(1 for game in recent_games if game['rewards']['coins_earned'] > 0)
            boost_count = sum(1 for game in recent_games if game['rewards']['boost_points'] > 0)
                            
            return {
                'win_rate': win_count / len(recent_games),
                'boost_rate': boost_count / len(recent_games)
            }
            
        except Exception as e:
            logging.error(f"Error calculating accuracy: {str(e)}")
            return {'win_rate': 0.0, 'boost_rate': 0.0}

    def _save_result(self, result: Dict):
        """Save individual prediction result"""
        try:
            filename = f"{self.results_dir}/game_{result['game_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error saving result: {str(e)}")

    def _save_progress(self):
        """Save current progress"""
        try:
            progress = {
                'current_coins': self.current_coins,
                'current_boost_points': self.current_boost_points,
                'learning_rate': self.learning_rate,
                'last_update': datetime.now().isoformat()
            }
            
            with open(f"{self.results_dir}/progress.json", 'w') as f:
                json.dump(progress, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error saving progress: {str(e)}")

    def _load_progress(self):
        """Load saved progress"""
        try:
            progress_file = f"{self.results_dir}/progress.json"
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.current_coins = progress.get('current_coins', 0)
                    self.current_boost_points = progress.get('current_boost_points', 0)
                    self.learning_rate = progress.get('learning_rate', 0.01)
                    
        except Exception as e:
            logging.error(f"Error loading progress: {str(e)}")

    def _adjust_learning_parameters(self):
        """Adjust learning parameters based on performance"""
        try:
            recent_accuracy = self._calculate_recent_accuracy()
            
            # Adjust learning rate based on performance
            if recent_accuracy['win_rate'] < 0.5:
                # Increase learning rate if performing poorly
                self.learning_rate = min(0.05, self.learning_rate * 1.1)
            else:
                # Decrease learning rate if performing well
                self.learning_rate = max(0.001, self.learning_rate * 0.95)
            
            # Calculate required performance for remaining games
            games_remaining = self.total_coins_goal - len(self.prediction_history)
            if games_remaining > 0:
                required_win_rate = (self.total_coins_goal - self.current_coins) / games_remaining
                required_boost_rate = (self.total_boost_goal - self.current_boost_points) / games_remaining
                
                logging.info(f"""
                Performance Adjustment:
                Required Win Rate: {required_win_rate:.2%}
                Required Boost Rate: {required_boost_rate:.2%}
                Current Learning Rate: {self.learning_rate}
                """)
                
        except Exception as e:
            logging.error(f"Error adjusting learning parameters: {str(e)}")

    def analyze_prediction_patterns(self) -> Dict:
        """Analyze patterns in prediction history"""
        try:
            if not self.prediction_history:
                return {}
                
            patterns = {
                'win_patterns': defaultdict(int),
                'score_patterns': defaultdict(list),
                'common_errors': defaultdict(int)
            }
            
            for entry in self.prediction_history[-50:]:  # Analyze last 50 games
                pred = entry['prediction']
                actual = entry['actual_result']
                
                # Analyze win/loss patterns
                if pred['prediction']['predicted_winner'] == actual['winner']:
                    patterns['win_patterns']['correct'] += 1
                else:
                    patterns['win_patterns']['incorrect'] += 1
                    
                # Analyze score prediction patterns
                score_diff = abs(pred['game_info']['score']['home'] - actual['home_score'])
                patterns['score_patterns']['differences'].append(score_diff)
                
                # Analyze error patterns
                if entry['analysis'].get('prediction_accuracy'):
                    if not entry['analysis']['prediction_accuracy']['winner_correct']:
                        patterns['common_errors']['wrong_winner'] += 1
                    if not entry['analysis']['prediction_accuracy']['within_margin']:
                        patterns['common_errors']['score_margin'] += 1
                        
            return self._summarize_patterns(patterns)
            
        except Exception as e:
            logging.error(f"Error analyzing prediction patterns: {str(e)}")
            return {}

    def _summarize_patterns(self, patterns: Dict) -> Dict:
        """Summarize analyzed patterns"""
        try:
            total_games = patterns['win_patterns']['correct'] + patterns['win_patterns']['incorrect']
            if total_games == 0:
                return {}
                
            return {
                'accuracy_metrics': {
                    'win_rate': patterns['win_patterns']['correct'] / total_games,
                    'average_score_diff': np.mean(patterns['score_patterns']['differences'])
                },
                'error_distribution': {
                    error_type: count / total_games
                    for error_type, count in patterns['common_errors'].items()
                },
                'recommendations': self._generate_recommendations(patterns, total_games)
            }
            
        except Exception as e:
            logging.error(f"Error summarizing patterns: {str(e)}")
            return {}

    def _generate_recommendations(self, patterns: Dict, total_games: int) -> List[str]:
        """Generate recommendations based on patterns"""
        recommendations = []
        
        try:
            # Win rate recommendations
            win_rate = patterns['win_patterns']['correct'] / total_games
            if win_rate < 0.5:
                recommendations.append("Focus on improving win/loss prediction accuracy")
                
            # Score prediction recommendations
            avg_score_diff = np.mean(patterns['score_patterns']['differences'])
            if avg_score_diff > self.score_margin_threshold:
                recommendations.append(
                    f"Improve score prediction accuracy (current avg diff: {avg_score_diff:.1f})"
                )
                
            # Error-specific recommendations
            for error_type, count in patterns['common_errors'].items():
                error_rate = count / total_games
                if error_rate > 0.3:  # If error occurs in more than 30% of games
                    recommendations.append(
                        f"Address frequent {error_type} errors (rate: {error_rate:.1%})"
                    )
                    
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations"]

