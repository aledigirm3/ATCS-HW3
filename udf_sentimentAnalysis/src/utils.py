import os
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from pathlib import Path

load_dotenv()

def get_project_root():
    """Returns the project root folder"""
    return Path(__file__).parent

def get_client():
    """Initialize and return Groq client"""
    groq_api_key = os.getenv('GROQ_API_KEY')
    return Groq(api_key=groq_api_key)

def get_model_name():
    """Get model name from environment"""
    return os.getenv('MODEL_NAME')

def load_data_files():
    """Load all data files needed for testing"""
    data_dir = get_project_root().parent / 'GT'
    return {
        "cards": pd.read_csv(data_dir / "cards_sample20.csv"),
        "rulings": pd.read_csv(data_dir / "ruling_sample.csv"),
        "posts": pd.read_csv(data_dir / "posts20sample.csv"),
        "users": pd.read_csv(data_dir / "users20sample.csv"),
        "user_reviews": pd.read_csv(data_dir / "googleplaystore_user_reviews_sample.csv")
    }