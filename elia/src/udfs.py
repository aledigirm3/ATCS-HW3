from typing import List
from utils import get_client, get_model_name

client = get_client()
model_name = get_model_name()

def analyze_sentiment(text: str) -> float:
    """Sentiment Analysis: Returns score between -1 (negative) and 1 (positive)"""
    if not text:
        return 0.0
    
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "Analyze sentiment of this text. Return ONLY a float between -1 (negative) and 1 (positive)."
        }, {
            "role": "user",
            "content": text
        }],
        model=model_name, 
        temperature=0.0,
        max_tokens=10
    )
    return float(chat_completion.choices[0].message.content)

def classify_entity(text: str, classes: List[str]) -> str:
    """Entity Classification: Categorizes text into predefined classes"""
    prompt = f"""Classify this text into one of {classes}:
    {text}
    Return ONLY the class name."""
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=0.0,
        max_tokens=20
    )
    return chat_completion.choices[0].message.content.strip()

def summarize(text: str, max_length: int = 50) -> str:
    """Summarization: Condenses text to specified length"""
    prompt = f"""Summarize this in under {max_length} characters:
    {text}
    Summary:"""
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=0.0,
        max_tokens=max_length
    )
    return chat_completion.choices[0].message.content.strip()
