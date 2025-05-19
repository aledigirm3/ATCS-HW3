import pandas as pd
from typing import List, Dict
from udfs import analyze_sentiment, classify_entity, summarize


def test_sentiment_analysis_001(cards: pd.DataFrame) -> List[str]:
    """
    Test Case ID: entity_classification_001
    Calculates the sentiment score on cards' flavourtext and returns the negative ones
    "Which cards have flavor text with negative sentiment?"
    equivalent SQL:
    SELECT name
    FROM cards
    WHERE flavorText IS NOT NULL
    AND analyze_sentiment(flavorText) < 0;
    """
    negative_cards = []
    for _, card in cards.iterrows():
        if pd.notna(card['flavorText']):
            sentiment = analyze_sentiment(card['flavorText'])
            if sentiment < 0:  # Negative sentiment
                negative_cards.append({"uuid":card["uuid"],"name":card['name']})
    return negative_cards

def test_sentiment_analysis_002(users: pd.DataFrame) -> List[str]:
    """
    Test Case ID: sentiment_analysis_002
    Finds users with AboutMe descriptions having a negative sentiment (score below -0.5)
    "Find users with AboutMe descriptions having a negative sentiment (score below -0.5)"
    equivalent SQL:
    SELECT Id
    FROM users
    WHERE AboutMe IS NOT NULL
    AND analyze_sentiment(AboutMe) < -0.5;
    """
    negative_users = []
    for _, user in users.iterrows():
        if pd.notna(user['AboutMe']):
            sentiment = analyze_sentiment(user['AboutMe'])
            if sentiment < -0.5:  # Highly negative sentiment
                negative_users.append(user['Id'])
    return negative_users

def test_sentiment_analysis_003(users: pd.DataFrame) -> List[str]:
    """
    Test Case ID: sentiment_analysis_003
    Finds user AboutMe descriptions with neutral sentiment (between -0.2 and 0.2)
    "Find user AboutMe descriptions with neutral sentiment (between -0.2 and 0.2)"
    equivalent SQL:
    SELECT Id
    FROM users
    WHERE AboutMe IS NOT NULL
    AND analyze_sentiment(AboutMe) BETWEEN -0.2 AND 0.2;
    """
    neutral_users = []
    for _, user in users.iterrows():
        if pd.notna(user['AboutMe']):
            sentiment = analyze_sentiment(user['AboutMe'])
            if -0.2 <= sentiment <= 0.2:  # Neutral sentiment
                neutral_users.append(user['Id'])
    return neutral_users

def test_sentiment_analysis_004(rulings: pd.DataFrame) -> List[str]:
    """
    Test Case ID: sentiment_analysis_004
    Finds rulings with sentiment outside the neutral range (-0.2 to 0.2)
    "Find rulings with sentiment outside the neutral range (-0.2 to 0.2)"
    equivalent SQL:
    SELECT id
    FROM rulings
    WHERE text IS NOT NULL
    AND (analyze_sentiment(text) < -0.2 OR analyze_sentiment(text) > 0.2);
    """
    distinctive_sentiment_rulings = []
    for _, ruling in rulings.iterrows():
        if pd.notna(ruling['text']):
            sentiment = analyze_sentiment(ruling['text'])
            if sentiment < -0.2 or sentiment > 0.2:  # Outside neutral range
                distinctive_sentiment_rulings.append(str(ruling['id']))
    return distinctive_sentiment_rulings


def test_sentiment_analysis_005(user_reviews: pd.DataFrame) -> List[int]:
    """
    Test Case ID: sentiment_analysis_005
    
    Calculates the sentiment score on user reviews and returns IDs of those with strongly negative sentiment
    
    "Which reviews have a negative sentiment score (below -0.3)?"
    
    equivalent SQL:
    SELECT id FROM user_reviews WHERE analyze_sentiment(Translated_Review) < -0.3;
    """
    negative_reviews = []
    for _, review in user_reviews.iterrows():
        if pd.notna(review['Translated_Review']):
            sentiment = analyze_sentiment(review['Translated_Review'])
            if sentiment < -0.3:  # Strongly negative sentiment
                negative_reviews.append(review['id'])
    return negative_reviews


def test_sentiment_analysis_006(user_reviews: pd.DataFrame) -> List[int]:
    """
    Test Case ID: sentiment_analysis_006
    
    Calculates the sentiment score on user reviews and returns IDs of those with highly positive sentiment
    
    "Find apps with reviews expressing highly positive sentiment (score above 0.6)"
    
    equivalent SQL:
    SELECT id FROM user_reviews WHERE analyze_sentiment(Translated_Review) > 0.6;
    """
    positive_reviews = []
    for _, review in user_reviews.iterrows():
        if pd.notna(review['Translated_Review']):
            sentiment = analyze_sentiment(review['Translated_Review'])
            if sentiment > 0.6:  # Highly positive sentiment
                positive_reviews.append(review['id'])
    return positive_reviews


def test_sentiment_analysis_007(user_reviews: pd.DataFrame) -> List[int]:
    """
    Test Case ID: sentiment_analysis_007
    
    Calculates the sentiment score on user reviews and returns IDs of those with neutral sentiment
    
    "Identify reviews with neutral sentiment (between -0.1 and 0.1)"
    
    equivalent SQL:
    SELECT id FROM user_reviews WHERE analyze_sentiment(Translated_Review) BETWEEN -0.1 AND 0.1;
    """
    neutral_reviews = []
    for _, review in user_reviews.iterrows():
        if pd.notna(review['Translated_Review']):
            sentiment = analyze_sentiment(review['Translated_Review'])
            if -0.1 <= sentiment <= 0.1:  # Neutral sentiment
                neutral_reviews.append(review['id'])
    return neutral_reviews

def test_sentiment_analysis_008(user_reviews: pd.DataFrame) -> dict:
    """
    Test Case ID: sentiment_analysis_008
    
    Identifies the review with the most negative sentiment in the dataset
    
    "What is the most negative review in the dataset?"
    
    equivalent SQL:
    SELECT id, App, Translated_Review, analyze_sentiment(Translated_Review) as sentiment_score 
    FROM user_reviews 
    WHERE Translated_Review IS NOT NULL
    ORDER BY analyze_sentiment(Translated_Review) ASC
    LIMIT 1;
    """
    most_negative_review = None
    lowest_sentiment_score = 0  # Start with neutral
    
    for _, review in user_reviews.iterrows():
        if pd.notna(review['Translated_Review']):
            sentiment = analyze_sentiment(review['Translated_Review'])
            if most_negative_review is None or sentiment < lowest_sentiment_score:
                lowest_sentiment_score = sentiment
                most_negative_review = review['id']
    
    return most_negative_review

def test_sentiment_analysis_classify_001(user_reviews: pd.DataFrame) -> dict:
    """
    Test Case ID: sentiment_analysis_classify_001
    
    Identifies the negative review about gaming apps
    
    "Which gaming app reviews have a negative sentiment score?"
    
    SELECT DISTINCT ur.id
    FROM user_reviews ur
    WHERE 
        classify_entity(ur.App, 'gaming,other') = 'gaming'
        AND analyze_sentiment(ur.Translated_Review) < 0;
    """
    gaming_apps = []
    
    for _, app in user_reviews.iterrows():
        # Classify the ruling text
        classification = classify_entity(
            text=app['App'],
            classes=["gaming", "other"],
        )
        
        if classification == "gaming" and analyze_sentiment(app["Translated_Review"])<0:
            gaming_apps.append(app['id'])
    
    return list(set(gaming_apps))

def test_analyze_sentiment_summarize_001(cards: pd.DataFrame, rulings: pd.DataFrame) -> List[Dict]:
    """
    Test Case ID: analyze_sentiment_summarize_001
    calls an llm to assing a sentiment score to flavour text of the cards,
    than takes the cards with positive score(>0.5)
    and finds those wid rulings, finally calls again the llm to produce
    a summary of said rulings
    "Cards with positive flavor text sentiment and complex rulings."
    equivalent SQL:
    SELECT
    c.name,
    summarize(string_agg(r.text, E'\n'), 100) AS summary,
    'positive' AS sentiment
    FROM cards c
    JOIN rulings r ON c.uuid = r.uuid
    WHERE c.flavorText IS NOT NULL
    AND analyze_sentiment(c.flavorText) > 0.5
    GROUP BY c.name;
    
    """
    complex_cards = []
    for _, card in cards.iterrows():
        if pd.notna(card['flavorText']):
            sentiment = analyze_sentiment(card['flavorText'])
            if sentiment > 0.5:
                card_rulings = rulings[rulings['uuid'] == card['uuid']]
                if not card_rulings.empty:
                    combined_rulings = "\n".join(card_rulings['text'])
                    summary = summarize(combined_rulings, 100)
                    complex_cards.append({
                        "uuid": card["uuid"],
                        "name": card['name'],
                        "summary": summary
                    })
    return complex_cards
