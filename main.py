# main.py

import asyncio
import logging
from datetime import datetime
import json
import os
from typing import Dict, Optional

from prediction_integrator import PredictionIntegrator
from base_predictor import BasePredictor

class PredictionSystem:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('prediction_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.base_predictor = BasePredictor()
        self.integrator = PredictionIntegrator(self.base_predictor)
        
        # Create results directory
        self.results_dir = "system_results"
        os.makedirs(self.results_dir, exist_ok=True)

    async def predict_game(self, game_info: Dict) -> Dict:
        """Make prediction for a game"""
        try:
            self.logger.info(f"Making prediction for game {game_info.get('id')}")
            
            # Get integrated prediction
            prediction = await self.integrator.predict_game(game_info)
            
            # Save prediction
            self._save_prediction(prediction)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return {'error': str(e)}

    async def update_game_result(self, game_id: str, actual_result: Dict):
        """Update system with actual game result"""
        try:
            self.logger.info(f"Updating system with result for game {game_id}")
            
            # Update integrator with result
            await self.integrator.update_with_game_result(game_id, actual_result)
            
            # Optimize prediction strategy
            await self.integrator.optimize_prediction_strategy()
            
            # Generate and save performance report
            await self._generate_performance_report()
            
        except Exception as e:
            self.logger.error(f"Error updating game result: {str(e)}")

    async def get_system_status(self) -> Dict:
        """Get current system status and performance metrics"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'performance_report': await self.integrator.generate_detailed_report(),
                'reward_progress': self.integrator.reward_system.get_progress_report(),
                'analysis_summary': self.integrator.analyzer.get_analysis_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}

    def _save_prediction(self, prediction: Dict):
        """Save prediction to file"""
        try:
            game_id = prediction['game_info']['id']
            filename = f"{self.results_dir}/prediction_{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(prediction, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving prediction: {str(e)}")

    async def _generate_performance_report(self):
        """Generate and save detailed performance report"""
        try:
            report = await self.integrator.generate_detailed_report()
            
            filename = f"{self.results_dir}/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")

# Example usage
async def main():
    # Initialize system
    system = PredictionSystem()
    
    # Example game info
    game_info = {
        'id': '12345',
        'home_team': 'Team A',
        'away_team': 'Team B',
        'date': '2024-01-01',
        'league': 'Example League'
    }
    
    # Make prediction
    prediction = await system.predict_game(game_info)
    print("Prediction:", json.dumps(prediction, indent=2))
    
    # Example game result
    actual_result = {
        'winner': 'Team A',
        'home_score': 3,
        'away_score': 1
    }
    
    # Update with result
    await system.update_game_result(game_info['id'], actual_result)
    
    # Get system status
    status = await system.get_system_status()
    print("System Status:", json.dumps(status, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
