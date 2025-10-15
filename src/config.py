"""
Configuration file for LLM Habituation Studies
Manages API credentials, model settings, and experimental parameters
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# API Keys - CREATE .ENV FILE AND ADD YOUR KEYS YOURSELF
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configurations
MODELS = {
    "claude": {
        "name": "claude-sonnet-4-5-20250929",
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": 1.0,  # Default, can override per study
    },
    "gpt4": {
        "name": "gpt-4-0125-preview",
        "provider": "openai",
        "max_tokens": 4096,
        "temperature": 1.0,
    },
    "gemini": {
        "name": "gemini-2.5-flash",  # Current stable model (October 2025)
        "provider": "google",
        "max_tokens": 4096,
        "temperature": 1.0,
    }
}

# Study parameters
STUDY_PARAMS = {
    "study_1": {
        "name": "habituation_induction",
        "n_repetitive": 100,
        "n_novel": 100,
        "pilot_n": 10,  # For testing
    },
    "study_2": {
        "name": "recovery_effect",
        "n_habituation": 30,
        "n_rest": 20,
        "n_retest": 20,
        "pilot_n": 5,
    },
    "study_3": {
        "name": "novelty_gradient",
        "n_conversations": 30,
        "turns_per_conversation": 10,
        "pilot_n": 3,
    },
    "study_4": {
        "name": "tolerance_patterns",
        "n_sessions": 20,
        "trials_per_session": 50,
        "pilot_n": 3,
    }
}

# Data paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROMPTS_DIR = DATA_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PROMPTS_DIR, RESULTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Metrics configuration
METRICS_CONFIG = {
    "entropy": {
        "tokenizer": "word",  # or "char"
        "normalize": True,
    },
    "lexical_diversity": {
        "methods": ["ttr", "mtld"],
        "min_length": 50,  # Minimum tokens for reliable MTLD
    },
    "semantic_novelty": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "metric": "cosine",
    }
}

# Statistical parameters
STATS_CONFIG = {
    "alpha": 0.05,
    "power": 0.80,
    "effect_size": 0.5,  # Medium effect (Cohen's d)
    "multiple_comparisons": "bonferroni",  # or "fdr"
}

# Rate limiting (be nice to APIs)
RATE_LIMITS = {
    "anthropic": {
        "requests_per_minute": 50,
        "tokens_per_minute": 100000,
    },
    "openai": {
        "requests_per_minute": 500,
        "tokens_per_minute": 150000,
    },
    "google": {
        "requests_per_minute": 60,
        "tokens_per_minute": 120000,
    }
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(BASE_DIR / "experiment.log"),
            "mode": "a",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}

def validate_config():
    """Validate that required API keys are present"""
    missing_keys = []
    
    if not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    
    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}\n"
            "CREATE .ENV FILE AND ADD YOUR KEYS YOURSELF"
        )
    
    return True

if __name__ == "__main__":
    # Quick validation check
    try:
        validate_config()
        print("✅ Configuration valid - all API keys present")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")