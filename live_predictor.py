# live_predictor.py

import logging
from datetime import datetime
from typing import Dict, Any
import numpy as np
from prediction_service import NBAPredictor

class LiveGamePredictor:
    def __init__(self, base_predictor: NBAPredictor): 
        self.base_predictor = base_predictor
        self.game_cache = {}
        self.update_interval = 30  # seconds

    def predict_live_game(self, game_info: Dict) -> Dict[str, Any]:
        """Make and update predictions for a live game."""
        game_id = game_info['gameId']
        current_time = datetime.now()

        # Check if we need to update prediction
        if self._should_update_prediction(game_id, current_time):
            prediction = self._make_live_prediction(game_info)
            self.game_cache[game_id] = {
                'last_update': current_time,
                'prediction': prediction
            }
            return prediction
        
        return self.game_cache[game_id]['prediction']

    def _should_update_prediction(self, game_id: str, current_time: datetime) -> bool:
        """Determine if prediction should be updated."""
        if game_id not in self.game_cache:
            return True

        last_update = self.game_cache[game_id]['last_update']
        time_diff = (current_time - last_update).total_seconds()
        return time_diff >= self.update_interval

    def _make_live_prediction(self, game_info: Dict) -> Dict[str, Any]:
        """Generate live prediction with current game state."""
        try:
            # Get base prediction
            base_prediction, model_predictions = self.base_predictor.predict_game(
                game_info['home_stats'],
                game_info['away_stats']
            )

            # Calculate in-game adjustments
            momentum_factor = self._calculate_momentum(game_info)
            performance_factor = self._calculate_performance_factor(game_info)
            time_pressure = self._calculate_time_pressure(game_info)

            # Adjust prediction
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
        except Exception as e:
            logging.error(f"Error in make_live_prediction: {str(e)}")
            raise

    def _parse_game_clock(self, clock_str: str) -> float:
        """Convert game clock string to minutes remaining."""
        try:
            if not clock_str or clock_str == 'N/A':
                return 12.0
            minutes, seconds = map(float, clock_str.split(':'))
            return minutes + (seconds / 60)
        except Exception as e:
            logging.warning(f"Error parsing game clock: {str(e)}")
            return 12.0

    def _calculate_momentum(self, game_info: Dict) -> float:
        """Calculate momentum factor based on recent scoring."""
        try:
            home_scores = [int(score) for score in game_info['scores']['home']['linescore']]
            away_scores = [int(score) for score in game_info['scores']['away']['linescore']]
            
            # Look at last 2 periods or all periods if less than 2
            recent_home = sum(home_scores[-2:])
            recent_away = sum(away_scores[-2:])
            
            momentum = (recent_home - recent_away) / max(recent_home + recent_away, 1)
            return max(min(momentum, 1.0), -1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating momentum: {str(e)}")
            return 0.0

    def _calculate_performance_factor(self, game_info: Dict) -> float:
        """Calculate performance relative to season averages."""
        try:
            home_stats = game_info['home_stats']['statistics'][0]
            away_stats = game_info['away_stats']['statistics'][0]
            
            home_ppg = float(home_stats.get('points', 0))
            away_ppg = float(away_stats.get('points', 0))
            
            # Calculate points per period
            current_home_pace = game_info['home_score'] / max(game_info['current_period'], 1)
            current_away_pace = game_info['away_score'] / max(game_info['current_period'], 1)
            
            # Compare current pace to season average
            home_performance = current_home_pace / (home_ppg/4) if home_ppg > 0 else 1.0
            away_performance = current_away_pace / (away_ppg/4) if away_ppg > 0 else 1.0
            
            return max(min(home_performance - away_performance, 1.0), -1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating performance factor: {str(e)}")
            return 0.0

    def _calculate_time_pressure(self, game_info: Dict) -> float:
        """Calculate time pressure factor."""
        try:
            total_minutes = 48.0  # Regular game length
            current_minute = (game_info['current_period'] - 1) * 12
            
            if game_info['clock'] and game_info['clock'] != 'N/A':
                minutes, seconds = map(float, game_info['clock'].split(':'))
                current_minute += (12 - minutes - seconds/60)
            
            return min(current_minute / total_minutes, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating time pressure: {str(e)}")
            return 0.0

    def _adjust_prediction(
        self,
        base_pred: float,
        momentum: float,
        performance: float,
        time_pressure: float,
        game_info: Dict
    ) -> float:
        """Combine factors to adjust prediction."""
        try:
            # Dynamic weights based on time pressure
            momentum_weight = 0.2 * (1 - time_pressure) + 0.3 * time_pressure
            performance_weight = 0.3 * (1 - time_pressure) + 0.2 * time_pressure
            score_weight = 0.5 * time_pressure  # Score becomes more important late game
            
            # Calculate score impact
            score_diff = game_info['home_score'] - game_info['away_score']
            max_diff = 20.0  # Maximum score difference to consider
            score_factor = max(min(score_diff / max_diff, 1.0), -1.0)
            
            # Combine factors with dynamic weights
            adjustment = (
                momentum * momentum_weight +
                performance * performance_weight +
                score_factor * score_weight
            )
            
            # Apply adjustment to base prediction
            adjusted_pred = base_pred + (adjustment * (1 - base_pred))
            
            # Ensure prediction stays between 0 and 1
            return max(min(adjusted_pred, 1.0), 0.0)
            
        except Exception as e:
            logging.warning(f"Error adjusting prediction: {str(e)}")
            return base_pred

    def get_prediction_confidence(self, prediction: Dict) -> float:
        """Calculate confidence level of the prediction."""
        try:
            # Get model agreement level
            predictions = list(prediction['model_predictions'].values())
            mean_pred = np.mean(predictions)
            std_dev = np.std(predictions)
            
            # Calculate model consensus (higher when models agree)
            model_consensus = 1 - (std_dev / 0.5)  # 0.5 is max possible std dev for probabilities
            
            # Factor in game state
            time_pressure = prediction['factors']['time_pressure']
            score_diff = abs(prediction['game_state']['score_difference'])
            
            # Confidence increases with:
            # - Higher model consensus
            # - Later in the game
            # - Larger score differences
            base_confidence = model_consensus * 0.4
            time_factor = time_pressure * 0.3
            score_factor = min(score_diff / 20.0, 1.0) * 0.3
            
            confidence = base_confidence + time_factor + score_factor
            
            return max(min(confidence, 1.0), 0.0)
            
        except Exception as e:
            logging.warning(f"Error calculating prediction confidence: {str(e)}")
            return 0.5

    def get_prediction_summary(self, game_info: Dict) -> Dict[str, Any]:
        """Generate a comprehensive prediction summary."""
        try:
            prediction = self.predict_live_game(game_info)
            confidence = self.get_prediction_confidence(prediction)
            
            return {
                'game_info': {
                    'home_team': game_info['home_team'],
                    'away_team': game_info['away_team'],
                    'current_score': f"{game_info['home_score']}-{game_info['away_score']}",
                    'period': game_info['current_period'],
                    'time_remaining': self._parse_game_clock(game_info['clock'])
                },
                'predictions': {
                    'base_probability': prediction['base_prediction'],
                    'adjusted_probability': prediction['adjusted_prediction'],
                    'confidence': confidence,
                    'model_predictions': prediction['model_predictions']
                },
                'factors': {
                    'momentum': prediction['factors']['momentum'],
                    'performance': prediction['factors']['performance'],
                    'time_pressure': prediction['factors']['time_pressure']
                },
                'recommendation': self._generate_recommendation(
                    prediction['adjusted_prediction'],
                    confidence,
                    game_info
                )
            }
            
        except Exception as e:
            logging.error(f"Error generating prediction summary: {str(e)}")
            raise

    def _generate_recommendation(
        self,
        probability: float,
        confidence: float,
        game_info: Dict
    ) -> Dict[str, Any]:
        """Generate betting recommendation based on prediction."""
        try:
            threshold = 0.6  # Minimum probability for a recommendation
            min_confidence = 0.7  # Minimum confidence level
            
            if confidence < min_confidence:
                return {
                    'recommendation': 'NO_BET',
                    'reason': 'Insufficient confidence in prediction'
                }
            
            if probability > threshold:
                return {
                    'recommendation': 'HOME_WIN',
                    'confidence': confidence,
                    'probability': probability,
                    'team': game_info['home_team']
                }
            elif (1 - probability) > threshold:
                return {
                    'recommendation': 'AWAY_WIN',
                    'confidence': confidence,
                    'probability': 1 - probability,
                    'team': game_info['away_team']
                }
            else:
                return {
                    'recommendation': 'NO_BET',
                    'reason': 'No clear advantage detected'
                }
                
        except Exception as e:
            logging.warning(f"Error generating recommendation: {str(e)}")
            return {'recommendation': 'NO_BET', 'reason': 'Error in analysis'}

    def analyze_prediction_history(self, game_id: str) -> Dict[str, Any]:
        """Analyze prediction history for a game."""
        try:
            if game_id not in self.game_cache:
                return {}
                
            predictions = self.game_cache[game_id].get('prediction_history', [])
            if not predictions:
                return {}
                
            # Calculate prediction stability
            prob_history = [p['adjusted_prediction'] for p in predictions]
            stability = 1 - np.std(prob_history)
            
            # Analyze trend
            trend = 'STABLE'
            if len(prob_history) >= 3:
                recent_trend = prob_history[-3:]
                if all(x < y for x, y in zip(recent_trend, recent_trend[1:])):
                    trend = 'INCREASING'
                elif all(x > y for x, y in zip(recent_trend, recent_trend[1:])):
                    trend = 'DECREASING'
            
            return {
                'stability': stability,
                'trend': trend,
                'prediction_count': len(predictions),
                'average_probability': np.mean(prob_history),
                'probability_range': {
                    'min': min(prob_history),
                    'max': max(prob_history)
                }
            }
            
        except Exception as e:
            logging.error(f"Error analyzing prediction history: {str(e)}")
            return {}

    def reset_cache(self):
        """Reset the prediction cache."""
        self.game_cache = {}
        logging.info("Prediction cache reset")

    def update_cache_settings(self, update_interval: int = None):
        """Update cache settings."""
        if update_interval is not None:
            self.update_interval = max(10, min(update_interval, 300))  # Between 10 and 300 seconds
            logging.info(f"Update interval set to {self.update_interval} seconds")


