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
            "content": """You are a precise sentiment analysis tool. Analyze the sentiment of the provided text and return ONLY a single float number between -1.0 and 1.0.
            
Guidelines:
- Use the range -1.0 to 1.0 for sentiment scores
- Scores < -0.5 indicate highly negative sentiment
- Scores > 0.5 indicate highly positive sentiment
- Scores between -0.2 and 0.2 represent neutral sentiment
- Use values between these ranges to indicate moderate sentiment intensity
- Your response must contain ONLY the numerical value (e.g., 0.7, -0.3, etc.)
- Do not include any explanations, words, symbols, or additional whitespace
- The response must be a valid float that can be parsed by Python's float() function"""
        }, {
            "role": "user",
            "content": text
        }],
        model=model_name, 
        temperature=0.0,
        max_tokens=10
    )
    
    try:
        return float(chat_completion.choices[0].message.content.strip())
    except ValueError:
        # Fallback in case parsing fails
        return 0.0

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

#few shot version
def analyze_sentiment_few_shot(text: str) -> float:
    """Sentiment Analysis: Returns score between -1 (negative) and 1 (positive)"""
    if not text:
        return 0.0
    
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": """You are a precise sentiment analysis tool. Analyze the sentiment of the provided text and return ONLY a single float number between -1.0 and 1.0.
            
Guidelines:
- Use the range -1.0 to 1.0 for sentiment scores
- Scores < -0.5 indicate highly negative sentiment
- Scores > 0.5 indicate highly positive sentiment
- Scores between -0.2 and 0.2 represent neutral sentiment
- Use values between these ranges to indicate moderate sentiment intensity
- Your response must contain ONLY the numerical value (e.g., 0.7, -0.3, etc.)
- Do not include any explanations, words, symbols, or additional whitespace"""
        }, {
            "role": "user",
            "content": "I absolutely hate this product. It's the worst purchase I've ever made and I want a refund immediately!"
        }, {
            "role": "assistant",
            "content": "-0.8"
        }, {
            "role": "user",
            "content": "The weather is okay today, nothing special."
        }, {
            "role": "assistant",
            "content": "0.1"
        }, {
            "role": "user",
            "content": "I love this new phone! The camera is amazing and battery life is fantastic."
        }, {
            "role": "assistant",
            "content": "0.7"
        }, {
            "role": "user",
            "content": "This movie was slightly disappointing given all the hype around it."
        }, {
            "role": "assistant",
            "content": "-0.3"
        }, {
            "role": "user",
            "content": text
        }],
        model=model_name, 
        temperature=0.0,
        max_tokens=10
    )
    
    try:
        return float(chat_completion.choices[0].message.content.strip())
    except ValueError:
        # Fallback in case parsing fails
        return 0.0