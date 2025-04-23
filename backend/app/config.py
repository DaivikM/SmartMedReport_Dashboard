import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Application Settings
DEBUG = True
RATE_LIMIT = 1.0  # seconds between API calls

# Visualization Settings
COLOR_PALETTE = {
    'background': '#ffffff',
    'text': '#ffffff',
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
    'accent1': '#06b6d4',
    'accent2': '#10b981',
    'accent3': '#f59e0b',
    'danger': '#ef4444',
    'grid': '#374151',
    'font_family': 'Times New Roman'
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
}