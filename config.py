# config.py

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'saved_models'
LOGS_DIR = BASE_DIR / 'logs'
PREDICTIONS_DIR = BASE_DIR / 'predictions'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(exist_ok=True)

# API Configuration
API_CONFIG = {
    'BASE_URL': 'https://api-nba-v1.p.rapidapi.com',
    'API_KEY': '89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c',
    'REQUEST_TIMEOUT': 30,
    'RATE_LIMIT_PAUSE': 1.0,
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 5,  # Seconds between retries
    'ENDPOINTS': {
        'LIVE_GAMES': '/games/live/',
        'GAME_STATS': '/statistics/games/gameId/',
        'TEAM_STATS': '/teams/statistics',
        'PLAYERS': '/players/teamId/',
        'STANDINGS': '/standings/standard/'
    }
}

# Model Configuration
MODEL_CONFIG = {
    'MODEL_DIR': str(MODELS_DIR),
    'PREDICTION_THRESHOLD': 0.5,
    'MODEL_WEIGHTS': {
        'random_forest': 0.25,
        'xgboost': 0.25,
        'svm': 0.15,
        'lstm': 0.15,
        'gru': 0.10,
        'gatv2_tcn': 0.10
    },
    'MODEL_FILES': {
        'random_forest': 'random_forest_20241111_040330.joblib',
        'xgboost': 'xgboost_20241111_040330.joblib',
        'svm': 'svm_20241111_040330.joblib',
        'lstm': 'lstm_20241111_040330.h5',
        'gru': 'gru_20241111_040330.h5',
        'gatv2_tcn': 'gatv2_tcn_20241111_040330.h5',
        'scaler': 'scaler_20241111_040330.joblib'
    }
}

# Feature Configuration
FEATURE_CONFIG = {
    'ROLLING_WINDOWS': [3, 5, 10],
    'REQUIRED_FEATURES': [
        'avg_points_diff',
        'avg_fg_pct_diff',
        'avg_3p_pct_diff',
        'avg_ft_pct_diff',
        'avg_rebounds_diff',
        'avg_assists_diff',
        'avg_steals_diff',
        'avg_blocks_diff',
        'avg_turnovers_diff',
        'win_pct_diff'
    ],
    'ADVANCED_METRICS': [
        'offensive_rating',
        'defensive_rating',
        'net_rating',
        'pace',
        'true_shooting_pct'
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'nba_predictions.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Prediction Service Configuration
PREDICTION_CONFIG = {
    'UPDATE_INTERVAL': 60,  # Seconds between updates
    'CONFIDENCE_THRESHOLD': 0.65,  # Minimum confidence for predictions
    'SAVE_PREDICTIONS': True,
    'PREDICTION_FORMAT': {
        'timestamp': '',
        'game_id': '',
        'home_team': '',
        'away_team': '',
        'home_win_probability': 0.0,
        'away_win_probability': 0.0,
        'model_predictions': {},
        'game_state': {
            'period': 0,
            'clock': '',
            'home_score': 0,
            'away_score': 0
        },
        'confidence_level': 0.0
    }
}

# Team Statistics Configuration
TEAM_STATS_CONFIG = {
    'SEASON_WEIGHT': 0.7,  # Weight for season-long statistics
    'RECENT_WEIGHT': 0.3,  # Weight for recent game statistics
    'MIN_GAMES_REQUIRED': 5,
    'STAT_CATEGORIES': {
        'OFFENSIVE': [
            'points',
            'fgp',
            'tpp',
            'ftp',
            'assists',
            'offReb'
        ],
        'DEFENSIVE': [
            'defReb',
            'steals',
            'blocks',
            'fouls'
        ],
        'ADVANCED': [
            'offensive_rating',
            'defensive_rating',
            'net_rating',
            'pace',
            'true_shooting_pct'
        ]
    }
}

# Error Handling Configuration
ERROR_CONFIG = {
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 5,
    'ERROR_CODES': {
        'API_ERROR': 1000,
        'MODEL_ERROR': 2000,
        'DATA_ERROR': 3000,
        'CONFIG_ERROR': 4000
    },
    'ERROR_MESSAGES': {
        1000: 'API Error: Failed to fetch data',
        2000: 'Model Error: Failed to make prediction',
        3000: 'Data Error: Invalid or missing data',
        4000: 'Config Error: Missing or invalid configuration'
    }
}

# Cache Configuration
CACHE_CONFIG = {
    'ENABLED': True,
    'EXPIRE_AFTER': 300,  # Seconds
    'MAX_SIZE': 1000,
    'CACHE_DIR': str(DATA_DIR / 'cache')
}
