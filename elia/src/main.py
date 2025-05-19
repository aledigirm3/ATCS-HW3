import json
import os
from pathlib import Path
from sentiment_examples_tests import *
from utils import load_data_files

def get_project_root():
    """Returns the project root folder"""
    return Path(__file__).parent

def run_tests():
    # Load data
    data = load_data_files()

    test_functions = {
        "sentiment_analysis_001": lambda: test_sentiment_analysis_001(data["cards"]),
        "sentiment_analysis_002": lambda: test_sentiment_analysis_002(data["users"]),
        "sentiment_analysis_003": lambda: test_sentiment_analysis_003(data["users"]),
        "sentiment_analysis_004": lambda: test_sentiment_analysis_004(data["rulings"]),
        "sentiment_analysis_005": lambda: test_sentiment_analysis_005(data["user_reviews"]),
        "sentiment_analysis_006": lambda: test_sentiment_analysis_006(data["user_reviews"]),
        "sentiment_analysis_007": lambda: test_sentiment_analysis_007(data["user_reviews"]),
        "sentiment_analysis_008": lambda: test_sentiment_analysis_008(data["user_reviews"]),
        "sentiment_analysis_classify_001":  lambda: test_sentiment_analysis_classify_001(data["user_reviews"]),
        "analyze_sentiment_summarize_001":  lambda: test_analyze_sentiment_summarize_001(data["user_reviews"]),
        "analyze_sentiment_summarize_001": lambda: test_analyze_sentiment_summarize_001(data["cards"],data["rulings"])
    }
    
    test_results = []
    
    project_root = get_project_root()
    
    # Path to JSON file
    json_path = project_root.parent / 'GT' / 'sentiment_analysis_examples.json'
    
    # Load test cases
    with open(json_path, 'r') as f:
        test_cases = json.load(f)
    
    # Execute tests
    for case in test_cases:
        test_id = case['unique_id']
        if test_id in test_functions:
            actual = test_functions[test_id]()
            test_results.append({
                "test_id": test_id,
                "question": case['question'],
                "expected": case['expected_result'],
                "actual": actual,
            })
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    run_tests()