# prediction_integrator.py

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json
import os
import numpy as np

from reward_system import PredictionRewardSystem
from prediction_analyzer import PredictionAnalyzer

class PredictionIntegrator:
    def __init__(self, base_predictor: Any):
        self.base_predictor = base_predictor
        self.reward_system = PredictionRewardSystem()
        self.analyzer = PredictionAnalyzer()
        
        self.results_dir = "prediction_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def predict_game(self, game_info: Dict) -> Dict:
        """Make prediction with integrated analysis and rewards"""
        try:
            # Get base prediction
            base_prediction = await self.base_predictor.predict_game(game_info)
            
            # Enhance prediction with analysis
            enhanced_prediction = self._enhance_prediction(base_prediction, game_info)
            
            # Store prediction for later analysis
            self._store_prediction(enhanced_prediction)
            
            return enhanced_prediction
            
        except Exception as e:
            self.logger.error(f"Error in prediction integration: {str(e)}")
            return {'error': str(e)}

    def _enhance_prediction(self, base_prediction: Dict, game_info: Dict) -> Dict:
        """Enhance base prediction with additional analysis"""
        try:
            # Add confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(base_prediction)
            
            # Add historical pattern analysis
            pattern_analysis = self.analyzer.analyze_prediction_patterns()
            
            enhanced_prediction = {
                **base_prediction,
                'confidence_metrics': confidence_metrics,
                'pattern_analysis': pattern_analysis,
                'timestamp': datetime.now().isoformat(),
                'game_info': game_info
            }
            
            return enhanced_prediction
            
        except Exception as e:
            self.logger.error(f"Error enhancing prediction: {str(e)}")
            return base_prediction

    def _calculate_confidence_metrics(self, prediction: Dict) -> Dict:
        """Calculate detailed confidence metrics"""
        try:
            return {
                'model_agreement': self._calculate_model_agreement(prediction),
                'historical_accuracy': self._calculate_historical_accuracy(prediction),
                'situational_confidence': self._calculate_situational_confidence(prediction)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence metrics: {str(e)}")
            return {}

    def update_with_game_result(self, game_id: str, actual_result: Dict):
        """Update systems with actual game result"""
        try:
            # Get stored prediction
            prediction = self._load_prediction(game_id)
            if not prediction:
                self.logger.warning(f"No stored prediction found for game {game_id}")
                return
                
            # Update reward system
            self.reward_system.update_metrics(game_id, prediction, actual_result)
            
            # Update analyzer
            self.analyzer.analyze_prediction(prediction, actual_result, prediction['game_info'])
            
            # Store analysis results
            self._store_analysis_result(game_id, prediction, actual_result)
            
        except Exception as e:
            self.logger.error(f"Error updating with game result: {str(e)}")

    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        try:
            return {
                'reward_progress': self.reward_system.get_progress_report(),
                'analysis_summary': self.analyzer.get_analysis_summary(),
                'performance_metrics': self._calculate_performance_metrics(),
                'recommendations': self._generate_combined_recommendations()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            return {
                'prediction_accuracy': {
                    'win_loss': self._calculate_win_loss_accuracy(),
                    'score_prediction': self._calculate_score_accuracy(),
                    'trends': self._calculate_accuracy_trends()
                },
                'reward_efficiency': {
                    'coins_per_game': self._calculate_coins_per_game(),
                    'boost_points_rate': self._calculate_boost_points_rate(),
                    'efficiency_trend': self._calculate_efficiency_trend()
                },
                'learning_metrics': {
                    'model_improvement': self._calculate_model_improvement(),
                    'adaptation_rate': self._calculate_adaptation_rate()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def _calculate_win_loss_accuracy(self, window: int = 50) -> Dict:
        """Calculate win/loss prediction accuracy"""
        try:
            recent_predictions = self._get_recent_predictions(window)
            correct_predictions = sum(1 for p in recent_predictions 
                                   if p.get('accuracy_metrics', {}).get('winner_correct', False))
            
            return {
                'overall_accuracy': correct_predictions / len(recent_predictions) if recent_predictions else 0,
                'recent_trend': self._calculate_trend([p.get('accuracy_metrics', {}).get('winner_correct', False) 
                                                     for p in recent_predictions])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating win/loss accuracy: {str(e)}")
            return {'overall_accuracy': 0, 'recent_trend': 0}

    def _calculate_score_accuracy(self, window: int = 50) -> Dict:
        """Calculate score prediction accuracy metrics"""
        try:
            recent_predictions = self._get_recent_predictions(window)
            score_errors = [p.get('accuracy_metrics', {}).get('score_accuracy', {}).get('total_diff', 0) 
                          for p in recent_predictions]
            
            return {
                'average_error': np.mean(score_errors) if score_errors else 0,
                'error_std': np.std(score_errors) if score_errors else 0,
                'within_threshold': sum(1 for e in score_errors if e <= 3) / len(score_errors) if score_errors else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating score accuracy: {str(e)}")
            return {'average_error': 0, 'error_std': 0, 'within_threshold': 0}

    def _calculate_accuracy_trends(self) -> Dict:
        """Calculate trends in prediction accuracy"""
        try:
            windows = [10, 20, 50]  # Different time windows for trend analysis
            trends = {}
            
            for window in windows:
                trends[f'window_{window}'] = {
                    'win_loss': self._calculate_win_loss_accuracy(window)['recent_trend'],
                    'score': self._calculate_trend(
                        [p.get('accuracy_metrics', {}).get('score_accuracy', {}).get('total_diff', 0) 
                         for p in self._get_recent_predictions(window)]
                    )
                }
                
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy trends: {str(e)}")
            return {}

    def _calculate_coins_per_game(self, window: int = 50) -> float:
        """Calculate average coins earned per game"""
        try:
            recent_rewards = self._get_recent_rewards(window)
            return np.mean([r.get('coins_earned', 0) for r in recent_rewards]) if recent_rewards else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating coins per game: {str(e)}")
            return 0.0

    def _calculate_boost_points_rate(self, window: int = 50) -> float:
        """Calculate rate of earning boost points"""
        try:
            recent_rewards = self._get_recent_rewards(window)
            return np.mean([r.get('boost_points', 0) for r in recent_rewards]) if recent_rewards else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating boost points rate: {str(e)}")
            return 0.0

    def _calculate_efficiency_trend(self) -> Dict:
        """Calculate trend in reward efficiency"""
        try:
            recent_rewards = self._get_recent_rewards(50)
            
            return {
                'coins_trend': self._calculate_trend([r.get('coins_earned', 0) for r in recent_rewards]),
                'boost_points_trend': self._calculate_trend([r.get('boost_points', 0) for r in recent_rewards])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency trend: {str(e)}")
            return {'coins_trend': 0, 'boost_points_trend': 0}

    def _calculate_model_improvement(self) -> Dict:
        """Calculate model improvement metrics"""
        try:
            windows = [10, 30, 50]  # Different time windows for improvement analysis
            improvements = {}
            
            for window in windows:
                recent_predictions = self._get_recent_predictions(window)
                improvements[f'window_{window}'] = {
                    'accuracy_improvement': self._calculate_improvement_rate(
                        [p.get('accuracy_metrics', {}).get('winner_correct', False) for p in recent_predictions]
                    ),
                    'confidence_improvement': self._calculate_improvement_rate(
                        [p.get('confidence_metrics', {}).get('model_agreement', 0) for p in recent_predictions]
                    )
                }
                
            return improvements
            
        except Exception as e:
            self.logger.error(f"Error calculating model improvement: {str(e)}")
            return {}

    def _calculate_adaptation_rate(self) -> Dict:
        """Calculate system adaptation rate metrics"""
        try:
            recent_predictions = self._get_recent_predictions(50)
            
            return {
                'learning_rate': self.reward_system.learning_rate,
                'adaptation_success': self._calculate_adaptation_success(recent_predictions),
                'recovery_rate': self._calculate_recovery_rate(recent_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptation rate: {str(e)}")
            return {}

    def _generate_combined_recommendations(self) -> List[str]:
        """Generate combined recommendations from all systems"""
        try:
            recommendations = []
            
            # Get recommendations from reward system
            reward_recommendations = self.reward_system.analyze_prediction_patterns()
            recommendations.extend(reward_recommendations.get('recommendations', []))
            
            # Get recommendations from analyzer
            analyzer_recommendations = self.analyzer.generate_improvement_recommendations()
            recommendations.extend(analyzer_recommendations)
            
            # Add performance-based recommendations
            performance_metrics = self._calculate_performance_metrics()
            if performance_metrics:
                recommendations.extend(
                    self._generate_performance_recommendations(performance_metrics)
                )

            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error generating combined recommendations: {str(e)}")
            return ["Error generating recommendations"]

    def _generate_performance_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on performance metrics"""
        try:
            recommendations = []
            
            # Check win/loss accuracy
            win_loss_accuracy = metrics.get('prediction_accuracy', {}).get('win_loss', {}).get('overall_accuracy', 0)
            if win_loss_accuracy < 0.6:
                recommendations.append(
                    f"Improve win/loss prediction accuracy (currently {win_loss_accuracy:.1%})"
                )

            # Check score prediction accuracy
            score_accuracy = metrics.get('prediction_accuracy', {}).get('score_prediction', {})
            if score_accuracy.get('average_error', 0) > 5:
                recommendations.append(
                    f"Reduce score prediction error (currently {score_accuracy['average_error']:.1f} points)"
                )

            # Check reward efficiency
            coins_per_game = metrics.get('reward_efficiency', {}).get('coins_per_game', 0)
            if coins_per_game < 0.7:
                recommendations.append(
                    f"Improve coins earning rate (currently {coins_per_game:.2f} coins per game)"
                )

            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating performance recommendations: {str(e)}")
            return []

    def _store_prediction(self, prediction: Dict):
        """Store prediction for later analysis"""
        try:
            game_id = prediction['game_info']['id']
            filename = f"{self.results_dir}/prediction_{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(prediction, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error storing prediction: {str(e)}")

    def _load_prediction(self, game_id: str) -> Optional[Dict]:
        """Load stored prediction for a game"""
        try:
            # Find most recent prediction file for game_id
            prediction_files = [f for f in os.listdir(self.results_dir) 
                              if f.startswith(f"prediction_{game_id}_")]
            
            if not prediction_files:
                return None
                
            latest_file = max(prediction_files)
            
            with open(f"{self.results_dir}/{latest_file}", 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading prediction: {str(e)}")
            return None

    def _store_analysis_result(self, game_id: str, prediction: Dict, actual_result: Dict):
        """Store analysis result"""
        try:
            analysis_result = {
                'game_id': game_id,
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'actual_result': actual_result,
                'analysis': self.analyzer.analyze_prediction(prediction, actual_result, prediction['game_info']),
                'rewards': self.reward_system.get_progress_report()
            }
            
            filename = f"{self.results_dir}/analysis_{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(analysis_result, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error storing analysis result: {str(e)}")

    def _get_recent_predictions(self, window: int) -> List[Dict]:
        """Get recent predictions for analysis"""
        try:
            prediction_files = sorted([f for f in os.listdir(self.results_dir) 
                                    if f.startswith("prediction_")])[-window:]
            
            predictions = []
            for file in prediction_files:
                with open(f"{self.results_dir}/{file}", 'r') as f:
                    predictions.append(json.load(f))
                    
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting recent predictions: {str(e)}")
            return []

    def _get_recent_rewards(self, window: int) -> List[Dict]:
        """Get recent reward records"""
        try:
            analysis_files = sorted([f for f in os.listdir(self.results_dir) 
                                   if f.startswith("analysis_")])[-window:]
            
            rewards = []
            for file in analysis_files:
                with open(f"{self.results_dir}/{file}", 'r') as f:
                    analysis = json.load(f)
                    rewards.append(analysis.get('rewards', {}))
                    
            return rewards
            
        except Exception as e:
            self.logger.error(f"Error getting recent rewards: {str(e)}")
            return []

    def _calculate_trend(self, values: List[Any]) -> float:
        """Calculate trend in a series of values"""
        try:
            if not values or len(values) < 2:
                return 0.0
                
            x = np.arange(len(values))
            y = np.array(values)
            
            slope, _ = np.polyfit(x, y, 1)
            return slope
            
        except Exception as e:
            self.logger.error(f"Error calculating trend: {str(e)}")
            return 0.0

    def _calculate_improvement_rate(self, values: List[Any]) -> float:
        """Calculate improvement rate"""
        try:
            if not values or len(values) < 2:
                return 0.0
                
            first_half = np.mean(values[:len(values)//2])
            second_half = np.mean(values[len(values)//2:])
            
            if first_half == 0:
                return 0.0
                
            return (second_half - first_half) / first_half
            
        except Exception as e:
            self.logger.error(f"Error calculating improvement rate: {str(e)}")
            return 0.0

    def _calculate_adaptation_success(self, predictions: List[Dict]) -> float:
        """Calculate success rate of system adaptations"""
        try:
            if not predictions:
                return 0.0
                
            adaptation_successes = []
            for i in range(1, len(predictions)):
                prev_correct = predictions[i-1].get('accuracy_metrics', {}).get('winner_correct', False)
                curr_correct = predictions[i].get('accuracy_metrics', {}).get('winner_correct', False)
                
                if not prev_correct and curr_correct:
                    adaptation_successes.append(1)
                else:
                    adaptation_successes.append(0)
                    
            return np.mean(adaptation_successes) if adaptation_successes else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptation success: {str(e)}")
            return 0.0

    def _calculate_recovery_rate(self, predictions: List[Dict]) -> float:
        """Calculate system's rate of recovery from errors"""
        try:
            if not predictions:
                return 0.0
                
            error_sequences = []
            current_sequence = 0
            
            for pred in predictions:
                if not pred.get('accuracy_metrics', {}).get('winner_correct', False):
                    current_sequence += 1
                else:
                    if current_sequence > 0:
                        error_sequences.append(current_sequence)
                    current_sequence = 0
            if error_sequences:
                return 1 / np.mean(error_sequences)
            return 1.0  # Perfect recovery rate if no errors
            
        except Exception as e:
            self.logger.error(f"Error calculating recovery rate: {str(e)}")
            return 0.0

    async def optimize_prediction_strategy(self):
        """Optimize prediction strategy based on historical performance"""
        try:
            # Get recent performance data
            performance_metrics = self._calculate_performance_metrics()
            analysis_summary = self.analyzer.get_analysis_summary()
            
            # Adjust prediction parameters
            adjustments = self._calculate_strategy_adjustments(
                performance_metrics,
                analysis_summary
            )
            
            # Apply adjustments to base predictor
            await self._apply_strategy_adjustments(adjustments)
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error optimizing prediction strategy: {str(e)}")
            return {}

    def _calculate_strategy_adjustments(self, performance_metrics: Dict, analysis_summary: Dict) -> Dict:
        """Calculate necessary strategy adjustments"""
        try:
            adjustments = {
                'confidence_threshold': self._adjust_confidence_threshold(performance_metrics),
                'feature_weights': self._adjust_feature_weights(analysis_summary),
                'model_weights': self._adjust_model_weights(performance_metrics),
                'learning_parameters': self._adjust_learning_parameters(performance_metrics)
            }
            
            self.logger.info(f"Calculated strategy adjustments: {adjustments}")
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy adjustments: {str(e)}")
            return {}

    def _adjust_confidence_threshold(self, performance_metrics: Dict) -> float:
        """Adjust confidence threshold based on performance"""
        try:
            current_accuracy = performance_metrics.get('prediction_accuracy', {}).get('win_loss', {}).get('overall_accuracy', 0)
            current_threshold = self.base_predictor.confidence_threshold
            
            if current_accuracy < 0.6:
                # Increase threshold if accuracy is low
                return min(0.9, current_threshold + 0.05)
            elif current_accuracy > 0.7:
                # Decrease threshold if accuracy is high
                return max(0.5, current_threshold - 0.05)
            
            return current_threshold
            
        except Exception as e:
            self.logger.error(f"Error adjusting confidence threshold: {str(e)}")
            return 0.7  # Default threshold

    def _adjust_feature_weights(self, analysis_summary: Dict) -> Dict:
        """Adjust feature weights based on importance"""
        try:
            feature_importance = self.analyzer.get_feature_importance_summary()
            current_weights = self.base_predictor.feature_weights
            
            adjusted_weights = {}
            total_importance = sum(f['average_importance'] for f in feature_importance.values())
            
            for feature, metrics in feature_importance.items():
                if total_importance > 0:
                    adjusted_weights[feature] = (
                        0.7 * current_weights.get(feature, 1.0) +
                        0.3 * (metrics['average_importance'] / total_importance)
                    )
                else:
                    adjusted_weights[feature] = current_weights.get(feature, 1.0)
                    
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"Error adjusting feature weights: {str(e)}")
            return self.base_predictor.feature_weights

    def _adjust_model_weights(self, performance_metrics: Dict) -> Dict:
        """Adjust model weights based on performance"""
        try:
            model_performance = performance_metrics.get('learning_metrics', {}).get('model_improvement', {})
            current_weights = self.base_predictor.model_weights
            
            adjusted_weights = {}
            for model, weight in current_weights.items():
                model_metrics = model_performance.get(f'window_50', {}).get(model, {})
                
                if model_metrics:
                    performance_factor = model_metrics.get('accuracy_improvement', 0)
                    adjusted_weights[model] = max(0.1, min(0.5, weight * (1 + performance_factor)))
                else:
                    adjusted_weights[model] = weight
                    
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            return {model: w/total_weight for model, w in adjusted_weights.items()}
            
        except Exception as e:
            self.logger.error(f"Error adjusting model weights: {str(e)}")
            return self.base_predictor.model_weights

    def _adjust_learning_parameters(self, performance_metrics: Dict) -> Dict:
        """Adjust learning parameters based on performance"""
        try:
            adaptation_rate = performance_metrics.get('learning_metrics', {}).get('adaptation_rate', {})
            
            return {
                'learning_rate': self._adjust_learning_rate(adaptation_rate),
                'momentum': self._adjust_momentum(adaptation_rate),
                'batch_size': self._adjust_batch_size(performance_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error adjusting learning parameters: {str(e)}")
            return {}

    def _adjust_learning_rate(self, adaptation_rate: Dict) -> float:
        """Adjust learning rate based on adaptation metrics"""
        try:
            current_rate = self.reward_system.learning_rate
            adaptation_success = adaptation_rate.get('adaptation_success', 0)
            
            if adaptation_success < 0.3:
                return min(0.05, current_rate * 1.2)  # Increase learning rate
            elif adaptation_success > 0.7:
                return max(0.001, current_rate * 0.8)  # Decrease learning rate
                
            return current_rate
            
        except Exception as e:
            self.logger.error(f"Error adjusting learning rate: {str(e)}")
            return 0.01

    def _adjust_momentum(self, adaptation_rate: Dict) -> float:
        """Adjust momentum parameter based on adaptation metrics"""
        try:
            recovery_rate = adaptation_rate.get('recovery_rate', 0)
            
            if recovery_rate < 0.3:
                return 0.9  # High momentum for slow recovery
            elif recovery_rate > 0.7:
                return 0.5  # Low momentum for quick recovery
                
            return 0.7  # Default momentum
            
        except Exception as e:
            self.logger.error(f"Error adjusting momentum: {str(e)}")
            return 0.7

    def _adjust_batch_size(self, performance_metrics: Dict) -> int:
        """Adjust batch size based on performance metrics"""
        try:
            accuracy_trend = performance_metrics.get('prediction_accuracy', {}).get('win_loss', {}).get('recent_trend', 0)
            
            if accuracy_trend < -0.1:
                return 5  # Smaller batch size for faster adaptation
            elif accuracy_trend > 0.1:
                return 20  # Larger batch size for stability
                
            return 10  # Default batch size
            
        except Exception as e:
            self.logger.error(f"Error adjusting batch size: {str(e)}")
            return 10

    async def _apply_strategy_adjustments(self, adjustments: Dict):
        """Apply calculated adjustments to the prediction system"""
        try:
            # Apply confidence threshold adjustment
            if 'confidence_threshold' in adjustments:
                self.base_predictor.confidence_threshold = adjustments['confidence_threshold']
                
            # Apply feature weight adjustments
            if 'feature_weights' in adjustments:
                self.base_predictor.feature_weights = adjustments['feature_weights']

            # Apply model weight adjustments
            if 'model_weights' in adjustments:
                self.base_predictor.model_weights = adjustments['model_weights']
                
            # Apply learning parameter adjustments
            if 'learning_parameters' in adjustments:
                await self._apply_learning_parameters(adjustments['learning_parameters'])
                
            self.logger.info("Successfully applied strategy adjustments")
            
        except Exception as e:
            self.logger.error(f"Error applying strategy adjustments: {str(e)}")

    async def _apply_learning_parameters(self, parameters: Dict):
        """Apply learning parameter adjustments"""
        try:
            if 'learning_rate' in parameters:
                self.reward_system.learning_rate = parameters['learning_rate']
                
            if 'momentum' in parameters:
                self.base_predictor.momentum = parameters['momentum']
                
            if 'batch_size' in parameters:
                self.base_predictor.batch_size = parameters['batch_size']
                
        except Exception as e:
            self.logger.error(f"Error applying learning parameters: {str(e)}")

    async def generate_detailed_report(self) -> Dict:
        """Generate detailed system performance report"""
        try:
            current_time = datetime.now()
            
            report = {
                'timestamp': current_time.isoformat(),
                'overall_performance': {
                    'reward_progress': self.reward_system.get_progress_report(),
                    'prediction_metrics': self._calculate_performance_metrics(),
                    'learning_effectiveness': await self._evaluate_learning_effectiveness()
                },
                'detailed_analysis': {
                    'pattern_analysis': self.analyzer.get_analysis_summary(),
                    'feature_importance': self.analyzer.get_feature_importance_summary(),
                    'error_analysis': await self._generate_error_analysis()
                },
                'optimization_status': {
                    'current_parameters': self._get_current_parameters(),
                    'recent_adjustments': await self._get_recent_adjustments(),
                    'adaptation_metrics': self._calculate_adaptation_metrics()
                },
                'recommendations': {
                    'immediate_actions': self._generate_immediate_recommendations(),
                    'long_term_improvements': self._generate_long_term_recommendations(),
                    'parameter_adjustments': await self._recommend_parameter_adjustments()
                }
            }
            
            # Save detailed report
            self._save_detailed_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating detailed report: {str(e)}")
            return {'error': str(e)}

    async def _evaluate_learning_effectiveness(self) -> Dict:
        """Evaluate effectiveness of the learning system"""
        try:
            recent_predictions = self._get_recent_predictions(50)
            
            return {
                'learning_curve': self._calculate_learning_curve(recent_predictions),
                'adaptation_speed': self._calculate_adaptation_speed(recent_predictions),
                'stability_metrics': self._calculate_stability_metrics(recent_predictions),
                'improvement_rates': {
                    'short_term': self._calculate_improvement_rate(recent_predictions[-10:]),
                    'medium_term': self._calculate_improvement_rate(recent_predictions[-30:]),
                    'long_term': self._calculate_improvement_rate(recent_predictions)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating learning effectiveness: {str(e)}")
            return {}

    async def _generate_error_analysis(self) -> Dict:
        """Generate detailed error analysis"""
        try:
            recent_errors = [p for p in self._get_recent_predictions(50) 
                           if not p.get('accuracy_metrics', {}).get('winner_correct', False)]
            
            return {
                'error_patterns': self._analyze_error_patterns(recent_errors),
                'error_distribution': self._calculate_error_distribution(recent_errors),
                'recovery_analysis': self._analyze_error_recovery(recent_errors),
                'impact_analysis': self._analyze_error_impact(recent_errors)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating error analysis: {str(e)}")
            return {}

    def _get_current_parameters(self) -> Dict:
        """Get current system parameters"""
        try:
            return {
                'prediction_parameters': {
                    'confidence_threshold': self.base_predictor.confidence_threshold,
                    'feature_weights': self.base_predictor.feature_weights,
                    'model_weights': self.base_predictor.model_weights
                },
                'learning_parameters': {
                    'learning_rate': self.reward_system.learning_rate,
                    'momentum': getattr(self.base_predictor, 'momentum', 0.7),
                    'batch_size': getattr(self.base_predictor, 'batch_size', 10)
                },
                'analysis_parameters': {
                    'score_threshold': self.analyzer.analysis_params['score_threshold'],
                    'confidence_threshold': self.analyzer.analysis_params['confidence_threshold'],
                    'pattern_detection_window': self.analyzer.analysis_params['pattern_detection_window']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current parameters: {str(e)}")
            return {}

    async def _get_recent_adjustments(self) -> List[Dict]:
        """Get history of recent parameter adjustments"""
        try:
            adjustment_files = sorted([f for f in os.listdir(self.results_dir) 
                                    if f.startswith("adjustment_")])[-10:]
            
            adjustments = []
            for file in adjustment_files:
                with open(f"{self.results_dir}/{file}", 'r') as f:
                    adjustments.append(json.load(f))
                    
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error getting recent adjustments: {str(e)}")
            return []

    def _calculate_adaptation_metrics(self) -> Dict:
        """Calculate system adaptation metrics"""
        try:
            recent_predictions = self._get_recent_predictions(50)
            
            return {
                'adaptation_speed': self._calculate_adaptation_speed(recent_predictions),
                'stability_index': self._calculate_stability_index(recent_predictions),
                'recovery_efficiency': self._calculate_recovery_efficiency(recent_predictions),
                'learning_consistency': self._calculate_learning_consistency(recent_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptation metrics: {str(e)}")
            return {}

    def _generate_immediate_recommendations(self) -> List[str]:
        """Generate recommendations for immediate actions"""
        try:
            recommendations = []
            performance_metrics = self._calculate_performance_metrics()
            
            # Check win/loss accuracy
            win_loss_accuracy = performance_metrics.get('prediction_accuracy', {}).get('win_loss', {}).get('overall_accuracy', 0)
            if win_loss_accuracy < 0.6:
                recommendations.append(
                    "URGENT: Improve win/loss prediction accuracy - consider adjusting confidence threshold"
                )
                
            # Check reward efficiency
            coins_per_game = performance_metrics.get('reward_efficiency', {}).get('coins_per_game', 0)
            if coins_per_game < 0.5:
                recommendations.append(
                    "URGENT: Boost reward efficiency - review prediction strategy for high-confidence games"
                )
                
            # Add more immediate recommendations based on other metrics...
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating immediate recommendations: {str(e)}")
            return ["Error generating recommendations"]


    def _generate_long_term_recommendations(self) -> List[str]:
        """Generate recommendations for long-term improvements"""
        try:
            recommendations = []
            analysis_summary = self.analyzer.get_analysis_summary()
            performance_trends = self._calculate_performance_metrics().get('prediction_accuracy', {}).get('trends', {})
            
            # Analyze long-term accuracy trends
            if performance_trends.get('window_50', {}).get('win_loss', 0) < 0:
                recommendations.append(
                    "Consider retraining base models with recent game data to improve accuracy"
                )
                
            # Analyze feature importance trends
            feature_importance = self.analyzer.get_feature_importance_summary()
            for feature, metrics in feature_importance.items():
                if metrics.get('trend', 0) < -0.1:
                    recommendations.append(
                        f"Investigate declining importance of feature '{feature}' and consider feature engineering"
                    )
                    
            # Analyze system adaptation
            adaptation_metrics = self._calculate_adaptation_metrics()
            if adaptation_metrics.get('learning_consistency', 0) < 0.6:
                recommendations.append(
                    "Implement more robust learning mechanisms to improve consistency"
                )
                
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating long-term recommendations: {str(e)}")
            return ["Error generating long-term recommendations"]

    async def _recommend_parameter_adjustments(self) -> Dict:
        """Recommend specific parameter adjustments"""
        try:
            current_params = self._get_current_parameters()
            performance_metrics = self._calculate_performance_metrics()
            
            recommendations = {
                'confidence_threshold': self._recommend_confidence_adjustment(
                    current_params['prediction_parameters']['confidence_threshold'],
                    performance_metrics
                ),
                'feature_weights': self._recommend_feature_weight_adjustments(
                    current_params['prediction_parameters']['feature_weights']
                ),
                'learning_rate': self._recommend_learning_rate_adjustment(
                    current_params['learning_parameters']['learning_rate'],
                    performance_metrics
                ),
                'batch_size': self._recommend_batch_size_adjustment(
                    current_params['learning_parameters']['batch_size'],
                    performance_metrics
                )
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error recommending parameter adjustments: {str(e)}")
            return {}

    def _recommend_confidence_adjustment(self, current_threshold: float, metrics: Dict) -> Dict:
        """Recommend confidence threshold adjustment"""
        try:
            win_rate = metrics.get('prediction_accuracy', {}).get('win_loss', {}).get('overall_accuracy', 0)
            boost_rate = metrics.get('reward_efficiency', {}).get('boost_points_rate', 0)
            
            recommendation = {
                'current_value': current_threshold,
                'recommended_value': current_threshold,
                'reason': "No adjustment needed"
            }
            
            if win_rate < 0.6 and boost_rate < 0.3:
                recommendation.update({
                    'recommended_value': min(0.9, current_threshold + 0.05),
                    'reason': "Increase threshold to improve accuracy and boost points"
                })
            elif win_rate > 0.7 and boost_rate > 0.4:
                recommendation.update({
                    'recommended_value': max(0.5, current_threshold - 0.05),
                    'reason': "Decrease threshold to maintain balance"
                })
                
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error recommending confidence adjustment: {str(e)}")
            return {'error': str(e)}

    def _recommend_feature_weight_adjustments(self, current_weights: Dict) -> Dict:
        """Recommend feature weight adjustments"""
        try:
            feature_importance = self.analyzer.get_feature_importance_summary()
            recommendations = {}
            
            for feature, current_weight in current_weights.items():
                importance_metrics = feature_importance.get(feature, {})
                avg_importance = importance_metrics.get('average_importance', 0)
                
                if avg_importance > current_weight + 0.1:
                    recommendations[feature] = {
                        'current_weight': current_weight,
                        'recommended_weight': min(1.0, current_weight + 0.1),
                        'reason': "Increase weight due to high importance"
                    }
                elif avg_importance < current_weight - 0.1:
                    recommendations[feature] = {
                        'current_weight': current_weight,
                        'recommended_weight': max(0.1, current_weight - 0.1),
                        'reason': "Decrease weight due to low importance"
                    }
                    
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error recommending feature weight adjustments: {str(e)}")
            return {}

    def _recommend_learning_rate_adjustment(self, current_rate: float, metrics: Dict) -> Dict:
        """Recommend learning rate adjustment"""
        try:
            adaptation_rate = metrics.get('learning_metrics', {}).get('adaptation_rate', {})
            adaptation_success = adaptation_rate.get('adaptation_success', 0)
            
            recommendation = {
                'current_value': current_rate,
                'recommended_value': current_rate,
                'reason': "No adjustment needed"
            }
            
            if adaptation_success < 0.3:
                recommendation.update({
                    'recommended_value': min(0.05, current_rate * 1.2),
                    'reason': "Increase learning rate to improve adaptation"
                })
            elif adaptation_success > 0.7:
                recommendation.update({
                    'recommended_value': max(0.001, current_rate * 0.8),
                    'reason': "Decrease learning rate to maintain stability"
                })
                
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error recommending learning rate adjustment: {str(e)}")
            return {'error': str(e)}

    def _recommend_batch_size_adjustment(self, current_size: int, metrics: Dict) -> Dict:
        """Recommend batch size adjustment"""
        try:
            accuracy_trend = metrics.get('prediction_accuracy', {}).get('win_loss', {}).get('recent_trend', 0)
            
            recommendation = {
                'current_value': current_size,
                'recommended_value': current_size,
                'reason': "No adjustment needed"
            }
            
            if accuracy_trend < -0.1:
                recommendation.update({
                    'recommended_value': max(5, current_size - 5),
                    'reason': "Decrease batch size for faster adaptation"
                })
            elif accuracy_trend > 0.1:
                recommendation.update({
                    'recommended_value': min(20, current_size + 5),
                    'reason': "Increase batch size for better stability"
                })
                
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error recommending batch size adjustment: {str(e)}")
            return {'error': str(e)}

    def _save_detailed_report(self, report: Dict):
        """Save detailed performance report"""
        try:
            filename = f"{self.results_dir}/detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4)
                
            self.logger.info(f"Saved detailed report to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving detailed report: {str(e)}")

