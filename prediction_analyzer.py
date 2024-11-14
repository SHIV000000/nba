# prediction_analyzer.py

from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os

class PredictionAnalyzer:
    def __init__(self):
        self.analysis_dir = "prediction_analysis"
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        self.error_patterns = defaultdict(list)
        self.success_patterns = defaultdict(list)
        self.feature_importance = {}
        
        # Initialize analysis parameters
        self.analysis_params = {
            'score_threshold': 3,  # Points margin for score prediction
            'confidence_threshold': 0.7,  # Minimum confidence for high-confidence predictions
            'pattern_detection_window': 10  # Games to analyze for patterns
        }

    def analyze_prediction(
        self,
        prediction: Dict,
        actual_result: Dict,
        game_stats: Dict
    ) -> Dict[str, Any]:
        """Comprehensive analysis of a single prediction"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'game_id': prediction['game_info']['id'],
                'accuracy_metrics': self._calculate_accuracy_metrics(prediction, actual_result),
                'error_analysis': self._analyze_errors(prediction, actual_result),
                'performance_factors': self._analyze_performance_factors(prediction, game_stats),
                'confidence_analysis': self._analyze_confidence(prediction, actual_result)
            }
            
            # Store analysis for pattern detection
            self._store_analysis_result(analysis)
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in prediction analysis: {str(e)}")
            return {'error': str(e)}

    def _calculate_accuracy_metrics(self, prediction: Dict, actual_result: Dict) -> Dict:
        """Calculate detailed accuracy metrics"""
        try:
            pred_winner = prediction['prediction']['predicted_winner']
            actual_winner = actual_result['winner']
            
            pred_score_home = prediction['game_info']['score']['home']
            pred_score_away = prediction['game_info']['score']['away']
            actual_score_home = actual_result['home_score']
            actual_score_away = actual_result['away_score']
            
            return {
                'winner_correct': pred_winner == actual_winner,
                'score_accuracy': {
                    'home_diff': abs(pred_score_home - actual_score_home),
                    'away_diff': abs(pred_score_away - actual_score_away),
                    'total_diff': abs(pred_score_home - actual_score_home) + 
                                abs(pred_score_away - actual_score_away),
                    'within_threshold': (
                        abs(pred_score_home - actual_score_home) <= self.analysis_params['score_threshold'] and
                        abs(pred_score_away - actual_score_away) <= self.analysis_params['score_threshold']
                    )
                },
                'margin_accuracy': {
                    'predicted_margin': pred_score_home - pred_score_away,
                    'actual_margin': actual_score_home - actual_score_away,
                    'margin_error': abs((pred_score_home - pred_score_away) - 
                                      (actual_score_home - actual_score_away))
                }
            }
            
        except Exception as e:
            logging.error(f"Error calculating accuracy metrics: {str(e)}")
            return {}

    def _analyze_errors(self, prediction: Dict, actual_result: Dict) -> Dict:
        """Analyze prediction errors and their potential causes"""
        try:
            errors = {
                'error_types': [],
                'error_factors': {},
                'severity': 'low'
            }
            
            # Check for winner prediction error
            if prediction['prediction']['predicted_winner'] != actual_result['winner']:
                errors['error_types'].append('incorrect_winner')
                errors['severity'] = 'high'
                
                # Analyze confidence in wrong prediction
                if prediction['prediction'].get('win_probability', 0) > 0.7:
                    errors['error_factors']['high_confidence_error'] = True
                    
            # Check score prediction errors
            score_diff_home = abs(prediction['game_info']['score']['home'] - actual_result['home_score'])
            score_diff_away = abs(prediction['game_info']['score']['away'] - actual_result['away_score'])
            
            if score_diff_home > self.analysis_params['score_threshold']:
                errors['error_types'].append('home_score_error')
                errors['error_factors']['home_score_diff'] = score_diff_home
                
            if score_diff_away > self.analysis_params['score_threshold']:
                errors['error_types'].append('away_score_error')
                errors['error_factors']['away_score_diff'] = score_diff_away
                
            # Calculate error severity
            if len(errors['error_types']) > 1:
                errors['severity'] = 'critical'
            elif score_diff_home + score_diff_away > 10:
                errors['severity'] = 'high'
                
            return errors
            
        except Exception as e:
            logging.error(f"Error in error analysis: {str(e)}")
            return {'error': str(e)}

    def _analyze_performance_factors(self, prediction: Dict, game_stats: Dict) -> Dict:
        """Analyze factors affecting prediction performance"""
        try:
            performance_factors = {
                'statistical_factors': {},
                'situational_factors': {},
                'model_agreement': {}
            }
            
            # Analyze statistical factors
            if 'model_predictions' in prediction['prediction']:
                model_preds = prediction['prediction']['model_predictions']
                performance_factors['model_agreement'] = {
                    'agreement_rate': self._calculate_model_agreement(model_preds),
                    'prediction_spread': self._calculate_prediction_spread(model_preds)
                }
                
            # Analyze game situation
            if 'period' in game_stats:
                performance_factors['situational_factors']['game_period'] = game_stats['period']
                performance_factors['situational_factors']['score_difference'] = (
                    game_stats.get('home_score', 0) - game_stats.get('away_score', 0)
                )
                
            return performance_factors
            
        except Exception as e:
            logging.error(f"Error in performance analysis: {str(e)}")
            return {}

    def _analyze_confidence(self, prediction: Dict, actual_result: Dict) -> Dict:
        """Analyze prediction confidence and its correlation with accuracy"""
        try:
            confidence_analysis = {
                'confidence_level': 'low',
                'confidence_justified': False,
                'confidence_metrics': {}
            }
            
            # Get prediction confidence
            win_probability = prediction['prediction'].get('win_probability', 0.5)
            
            # Determine confidence level
            if win_probability > 0.8:
                confidence_analysis['confidence_level'] = 'very_high'
            elif win_probability > 0.7:
                confidence_analysis['confidence_level'] = 'high'
            elif win_probability > 0.6:
                confidence_analysis['confidence_level'] = 'medium'
                
            # Check if confidence was justified
            prediction_correct = prediction['prediction']['predicted_winner'] == actual_result['winner']
            confidence_analysis['confidence_justified'] = (
                (win_probability > 0.7 and prediction_correct) or
                (win_probability < 0.6 and not prediction_correct)
            )
            
            # Calculate confidence metrics
            confidence_analysis['confidence_metrics'] = {
                'win_probability': win_probability,
                'actual_outcome': prediction_correct,
                'confidence_error': abs(win_probability - (1 if prediction_correct else 0))
            }
            
            return confidence_analysis
            
        except Exception as e:
            logging.error(f"Error in confidence analysis: {str(e)}")
            return {}

    def _calculate_model_agreement(self, model_predictions: Dict) -> float:
        """Calculate agreement rate between different models"""
        try:
            if not model_predictions:
                return 0.0
                
            predictions = list(model_predictions.values())
            majority_prediction = sum(1 for p in predictions if p > 0.5) > len(predictions) / 2
            agreement_count = sum(1 for p in predictions if (p > 0.5) == majority_prediction)
            
            return agreement_count / len(predictions)
            
        except Exception as e:
            logging.error(f"Error calculating model agreement: {str(e)}")
            return 0.0

    def _calculate_prediction_spread(self, model_predictions: Dict) -> float:
        """Calculate spread of predictions across different models"""
        try:
            if not model_predictions:
                return 0.0
                
            predictions = list(model_predictions.values())
            return np.std(predictions)
            
        except Exception as e:
            logging.error(f"Error calculating prediction spread: {str(e)}")
            return 0.0

    def _store_analysis_result(self, analysis: Dict):
        """Store analysis result for pattern detection"""
        try:
            filename = f"{self.analysis_dir}/analysis_{analysis['game_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=4)
                
            # Store in pattern detection system
            if analysis.get('error_analysis', {}).get('error_types'):
                self.error_patterns[analysis['game_id']] = analysis
            else:
                self.success_patterns[analysis['game_id']] = analysis
                
        except Exception as e:
            logging.error(f"Error storing analysis result: {str(e)}")

    def generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis patterns"""
        try:
            recommendations = []
            
            # Analyze recent error patterns
            recent_errors = list(self.error_patterns.values())[-self.analysis_params['pattern_detection_window']:]
            error_counts = defaultdict(int)
            
            for error in recent_errors:
                for error_type in error.get('error_analysis', {}).get('error_types', []):
                    error_counts[error_type] += 1
                    
            # Generate recommendations based on error patterns
            for error_type, count in error_counts.items():
                if count >= self.analysis_params['pattern_detection_window'] * 0.3:  # If error occurs in 30% of games
                    recommendations.append(
                        f"Focus on reducing {error_type} errors - occurred in {count} of last {self.analysis_params['pattern_detection_window']} games"
                    )
            
            # Analyze confidence patterns
            confidence_errors = self._analyze_confidence_patterns()
            if confidence_errors['high_confidence_errors'] > 0.3:
                recommendations.append(
                    "Adjust confidence calculation - high confidence predictions show significant error rate"
                )
                
            # Analyze score prediction patterns
            score_patterns = self._analyze_score_patterns()
            if score_patterns['average_error'] > self.analysis_params['score_threshold']:
                recommendations.append(
                    f"Improve score prediction accuracy - average error ({score_patterns['average_error']:.1f} points) exceeds threshold"
                )
                
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations"]

    def _analyze_confidence_patterns(self) -> Dict:
        """Analyze patterns in prediction confidence"""
        try:
            recent_analyses = list(self.error_patterns.values())[-self.analysis_params['pattern_detection_window']:] + \
                            list(self.success_patterns.values())[-self.analysis_params['pattern_detection_window']:]
            
            high_confidence_count = 0
            high_confidence_errors = 0
            
            for analysis in recent_analyses:
                confidence_analysis = analysis.get('confidence_analysis', {})
                if confidence_analysis.get('confidence_level') in ['high', 'very_high']:
                    high_confidence_count += 1
                    if not confidence_analysis.get('confidence_justified', False):
                        high_confidence_errors += 1
                        
            return {
                'high_confidence_errors': high_confidence_errors / max(high_confidence_count, 1),
                'high_confidence_predictions': high_confidence_count / len(recent_analyses) if recent_analyses else 0
            }
            
        except Exception as e:
            logging.error(f"Error analyzing confidence patterns: {str(e)}")
            return {'high_confidence_errors': 0, 'high_confidence_predictions': 0}

    def _analyze_score_patterns(self) -> Dict:
        """Analyze patterns in score predictions"""
        try:
            recent_analyses = list(self.error_patterns.values())[-self.analysis_params['pattern_detection_window']:] + \
                            list(self.success_patterns.values())[-self.analysis_params['pattern_detection_window']:]
            
            score_errors = []
            for analysis in recent_analyses:
                accuracy_metrics = analysis.get('accuracy_metrics', {}).get('score_accuracy', {})
                if 'total_diff' in accuracy_metrics:
                    score_errors.append(accuracy_metrics['total_diff'])
                    
            return {
                'average_error': np.mean(score_errors) if score_errors else 0,
                'error_std': np.std(score_errors) if score_errors else 0,
                'improvement_trend': self._calculate_improvement_trend(score_errors)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing score patterns: {str(e)}")
            return {'average_error': 0, 'error_std': 0, 'improvement_trend': 0}

    def _calculate_improvement_trend(self, errors: List[float]) -> float:
        """Calculate trend in prediction improvements"""
        try:
            if len(errors) < 2:
                return 0
                
            # Calculate moving average of errors
            window_size = min(5, len(errors) // 2)
            if window_size < 2:
                return 0
                
            recent_avg = np.mean(errors[-window_size:])
            previous_avg = np.mean(errors[:-window_size])
            
            # Return improvement percentage
            if previous_avg == 0:
                return 0
            return (previous_avg - recent_avg) / previous_avg
            
        except Exception as e:
            logging.error(f"Error calculating improvement trend: {str(e)}")
            return 0

    def get_analysis_summary(self) -> Dict:
        """Generate summary of recent prediction analysis"""
        try:
            recent_analyses = list(self.error_patterns.values())[-self.analysis_params['pattern_detection_window']:] + \
                            list(self.success_patterns.values())[-self.analysis_params['pattern_detection_window']:]
            
            if not recent_analyses:
                return {'error': 'No recent analyses available'}
                
            summary = {
                'period': {
                    'start': recent_analyses[0]['timestamp'],
                    'end': recent_analyses[-1]['timestamp'],
                    'games_analyzed': len(recent_analyses)
                },
                'accuracy_metrics': self._summarize_accuracy_metrics(recent_analyses),
                'error_patterns': self._summarize_error_patterns(recent_analyses),
                'confidence_metrics': self._summarize_confidence_metrics(recent_analyses),
                'recommendations': self.generate_improvement_recommendations()
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating analysis summary: {str(e)}")
            return {'error': str(e)}

    def _summarize_accuracy_metrics(self, analyses: List[Dict]) -> Dict:
        """Summarize accuracy metrics from multiple analyses"""
        try:
            correct_predictions = sum(1 for a in analyses 
                                   if a.get('accuracy_metrics', {}).get('winner_correct', False))
            score_accuracies = [a.get('accuracy_metrics', {}).get('score_accuracy', {}).get('total_diff', 0) 
                              for a in analyses]
            
            return {
                'win_prediction_accuracy': correct_predictions / len(analyses),
                'average_score_error': np.mean(score_accuracies),
                'score_error_std': np.std(score_accuracies),
                'improvement_trend': self._calculate_improvement_trend(score_accuracies)
            }
            
        except Exception as e:
            logging.error(f"Error summarizing accuracy metrics: {str(e)}")
            return {}

    def _summarize_error_patterns(self, analyses: List[Dict]) -> Dict:
        """Summarize error patterns from multiple analyses"""
        try:
            error_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            
            for analysis in analyses:
                error_analysis = analysis.get('error_analysis', {})
                for error_type in error_analysis.get('error_types', []):
                    error_counts[error_type] += 1
                severity_counts[error_analysis.get('severity', 'low')] += 1
                
            return {
                'common_errors': {
                    error: count/len(analyses) 
                    for error, count in error_counts.items()
                },
                'severity_distribution': {
                    severity: count/len(analyses) 
                    for severity, count in severity_counts.items()
                }
            }
            
        except Exception as e:
            logging.error(f"Error summarizing error patterns: {str(e)}")
            return {}

    def _summarize_confidence_metrics(self, analyses: List[Dict]) -> Dict:
        """Summarize confidence metrics from multiple analyses"""
        try:
            confidence_levels = defaultdict(int)
            justified_confidence = 0
            
            for analysis in analyses:
                conf_analysis = analysis.get('confidence_analysis', {})
                confidence_levels[conf_analysis.get('confidence_level', 'low')] += 1
                if conf_analysis.get('confidence_justified', False):
                    justified_confidence += 1
                    
            return {
                'confidence_distribution': {
                    level: count/len(analyses) 
                    for level, count in confidence_levels.items()
                },
                'confidence_reliability': {
                    'justified_confidence_rate': justified_confidence / len(analyses),
                    'confidence_correlation': self._calculate_confidence_correlation(analyses)
                }
            }
            
        except Exception as e:
            logging.error(f"Error summarizing confidence metrics: {str(e)}")
            return {}

    def _calculate_confidence_correlation(self, analyses: List[Dict]) -> float:
        """Calculate correlation between confidence and accuracy"""
        try:
            confidences = []
            accuracies = []
            
            for analysis in analyses:
                conf_metrics = analysis.get('confidence_analysis', {}).get('confidence_metrics', {})
                if 'win_probability' in conf_metrics and 'actual_outcome' in conf_metrics:
                    confidences.append(conf_metrics['win_probability'])
                    accuracies.append(float(conf_metrics['actual_outcome']))
                    
            if len(confidences) < 2:
                return 0.0
                
            return np.corrcoef(confidences, accuracies)[0, 1]
            
        except Exception as e:
            logging.error(f"Error calculating confidence correlation: {str(e)}")
            return 0.0

    def export_analysis_report(self, filepath: str = None) -> bool:
        """Export detailed analysis report to file"""
        try:
            if filepath is None:
                filepath = f"{self.analysis_dir}/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
            report = {
                'generated_at': datetime.now().isoformat(),
                'analysis_summary': self.get_analysis_summary(),
                'improvement_recommendations': self.generate_improvement_recommendations(),
                'detailed_metrics': {
                    'error_patterns': dict(self.error_patterns),
                    'success_patterns': dict(self.success_patterns),
                    'feature_importance': self.feature_importance
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=4)
                
            return True
            
        except Exception as e:
            logging.error(f"Error exporting analysis report: {str(e)}")
            return False

    def update_feature_importance(self, game_id: str, features: Dict[str, float]):
        """Update feature importance based on prediction results"""
        try:
            for feature, importance in features.items():
                if feature not in self.feature_importance:
                    self.feature_importance[feature] = []
                self.feature_importance[feature].append(importance)
                
            # Keep only recent feature importance values
            max_history = 100
            for feature in self.feature_importance:
                if len(self.feature_importance[feature]) > max_history:
                    self.feature_importance[feature] = self.feature_importance[feature][-max_history:]
                    
        except Exception as e:
            logging.error(f"Error updating feature importance: {str(e)}")

    def get_feature_importance_summary(self) -> Dict[str, float]:
        """Get summary of feature importance"""
        try:
            summary = {}
            for feature, values in self.feature_importance.items():
                if values:
                    summary[feature] = {
                        'average_importance': np.mean(values),
                        'importance_std': np.std(values),
                        'trend': self._calculate_improvement_trend(values)
                    }
            return summary
            
        except Exception as e:
            logging.error(f"Error getting feature importance summary: {str(e)}")
            return {}

    def reset_analysis(self):
        """Reset analysis data"""
        try:
            self.error_patterns.clear()
            self.success_patterns.clear()
            self.feature_importance.clear()
            logging.info("Analysis data reset successfully")
            
        except Exception as e:
            logging.error(f"Error resetting analysis data: {str(e)}")

