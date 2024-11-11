# api_client.py


import requests
import logging
from typing import Dict, List, Optional
from time import sleep
from datetime import datetime
import json

class EnhancedNBAApiClient:
    def __init__(self, api_key: str):
        self.headers = {
            'x-rapidapi-host': 'api-nba-v1.p.rapidapi.com',
            'x-rapidapi-key': api_key
        }
        self.base_url = 'https://api-nba-v1.p.rapidapi.com'

    def get_live_games(self) -> List[Dict]:
        """Get current live games."""
        endpoint = f"{self.base_url}/games"
        params = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'season': '2024',
            'league': 'standard'
        }
        
        response = self._make_request('GET', endpoint, params)
        all_games = response.get('response', [])
        
        # Filter for games that are in play
        live_games = [
            game for game in all_games 
            if game.get('status', {}).get('long') == "In Play" or
            game.get('status', {}).get('short') == 2  # Status code 2 indicates live games
        ]
        
        processed_games = []
        for game in live_games:
            processed_game = {
                'id': game.get('id'),
                'teams': {
                    'home': {
                        'id': game.get('teams', {}).get('home', {}).get('id'),
                        'name': game.get('teams', {}).get('home', {}).get('name'),
                    },
                    'away': {
                        'id': game.get('teams', {}).get('visitors', {}).get('id'),
                        'name': game.get('teams', {}).get('visitors', {}).get('name'),
                    }
                },
                'scores': {
                    'home': {
                        'points': int(game.get('scores', {}).get('home', {}).get('points', 0))
                    },
                    'away': {
                        'points': int(game.get('scores', {}).get('visitors', {}).get('points', 0))
                    }
                },
                'periods': {
                    'current': game.get('periods', {}).get('current', 0),
                    'total': game.get('periods', {}).get('total', 4)
                },
                'status': {
                    'clock': game.get('status', {}).get('clock', ''),
                    'long': game.get('status', {}).get('long', ''),
                }
            }
            processed_games.append(processed_game)
        
        return processed_games

    def get_team_stats(self, team_id: str) -> Dict:
        """Get team statistics with proper error handling."""
        # First try getting current season stats
        endpoint = f"{self.base_url}/teams/statistics"
        params = {
            'id': team_id,
            'season': '2024'  # Remove 'league' parameter as it's causing errors
        }
        
        try:
            response = self._make_request('GET', endpoint, params)
            logging.debug(f"Team stats response for ID {team_id}: {json.dumps(response, indent=2)}")
            
            # If current season stats aren't available, try previous season
            if not response.get('response') or response.get('errors'):
                logging.info(f"No current season stats for team {team_id}, trying previous season")
                params['season'] = '2023'
                response = self._make_request('GET', endpoint, params)
            
            stats = response.get('response', [])
            if not stats:
                logging.warning(f"No statistics found for team ID: {team_id}")
                return {}
                
            # Process stats using the new method
            return self.process_team_stats(stats[0])
            
        except Exception as e:
            logging.error(f"Error fetching team stats for ID {team_id}: {str(e)}")
            return {}

    def process_team_stats(self, stats: Dict) -> Dict:
        """Process raw team statistics into required format."""
        try:
            games = float(stats.get('games', 1))
            return {
                'statistics': [{
                    'points': float(stats.get('points', 0)) / games,
                    'fieldGoalsPercentage': float(stats.get('fgp', 0)),
                    'threePointsPercentage': float(stats.get('tpp', 0)),
                    'freeThrowsPercentage': float(stats.get('ftp', 0)),
                    'reboundsTotal': float(stats.get('totReb', 0)) / games,
                    'assists': float(stats.get('assists', 0)) / games,
                    'steals': float(stats.get('steals', 0)) / games,
                    'blocks': float(stats.get('blocks', 0)) / games,
                    'turnovers': float(stats.get('turnovers', 0)) / games,
                    'games': games,
                    'wins': float(stats.get('wins', 0))
                }]
            }
        except Exception as e:
            logging.error(f"Error processing team stats: {str(e)}")
            return {'statistics': [{}]}

    def get_game_statistics(self, game_id: str) -> Dict:
        """Get detailed game statistics."""
        endpoint = f"{self.base_url}/games/statistics"
        params = {
            'id': game_id
        }
        response = self._make_request('GET', endpoint, params)
        return response.get('response', {})

    def get_h2h(self, team1_id: str, team2_id: str, season: str = "2023") -> List[Dict]:
        """Get head-to-head matches between two teams."""
        endpoint = f"{self.base_url}/games"
        params = {
            'h2h': f"{team1_id}-{team2_id}",
            'season': season
        }
        response = self._make_request('GET', endpoint, params)
        return response.get('response', [])

    def get_team_stats_alternative(self, team_id: str) -> Dict:
        """Alternative method to get team statistics using standings endpoint."""
        endpoint = f"{self.base_url}/standings"
        params = {
            'team': team_id,
            'season': '2024'
        }
        
        try:
            response = self._make_request('GET', endpoint, params)
            standings = response.get('response', [])
            
            if not standings:
                return {}
                
            team_stats = standings[0]
            
            # Convert standings data to our statistics format
            processed_stats = self.process_team_stats(team_stats)
            
            return processed_stats
            
        except Exception as e:
            logging.error(f"Error fetching alternative team stats for ID {team_id}: {str(e)}")
            return {}

    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with retry logic."""
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method,
                    endpoint,
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    sleep(retry_delay)
                else:
                    raise

        return {}

