import json
import ast
from sklearn.metrics import precision_score, recall_score, f1_score
from bert_score import score
from utils import get_project_root
import os
import warnings

def parse_actual(actual):
    """Parse actual result from string or return as-is if already a list."""
    if isinstance(actual, str):
        try:
            parsed = ast.literal_eval(actual)
            return parsed if isinstance(parsed, list) else [parsed]
        except:
            return [actual]
    elif isinstance(actual, list):
        return actual
    else:
        return [actual]  # Handle single integers or other types

def evaluate_ids(expected, actual):
    """Calculate precision, recall, F1 for ID lists."""
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    if not expected or not actual:
        return 0.0, 0.0, 0.0
    
    expected_set = set(expected)
    actual_set = set(actual)
    
    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_summaries(expected, actual):
    """Calculate BERTScore for summaries."""
    # Match by ID
    expected_by_id = {item['id']: item.get('summary', '') for item in expected}
    actual_by_id = {}
    
    for item in actual:
        item_id = item['id']
        # Get the text from any field that's not 'id'
        text_fields = {k: v for k, v in item.items() if k != 'id'}
        if text_fields:
            # Use the first non-id field as the text
            actual_by_id[item_id] = list(text_fields.values())[0]
        else:
            actual_by_id[item_id] = ''
    
    expected_texts = []
    actual_texts = []
    
    for exp_id, exp_text in expected_by_id.items():
        if exp_id in actual_by_id:
            expected_texts.append(exp_text)
            actual_texts.append(actual_by_id[exp_id])
    
    if not expected_texts:
        return 0.0, 0.0, 0.0
    
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    P, R, F1 = score(actual_texts, expected_texts, lang='en', verbose=False)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def main():

    warnings.filterwarnings('ignore')
    json_path=get_project_root().parent/ 'Results' / 'test_results.json'
    with open(json_path, 'r') as f:
        tests = json.load(f)
    
    all_p, all_r, all_f1 = [], [], []

    print("Test Evaluation Results:")
    print("-" * 50)
    
    for test in tests:
        test_id = test['test_id']
        expected = test['expected']
        actual = parse_actual(test['actual'])
        
         # Check if it's a summary test (contains dicts)
        if expected and isinstance(expected[0], dict) and 'summary' in expected[0]:
            p, r, f1 = evaluate_summaries(expected, actual)
        else:
            p, r, f1 = evaluate_ids(expected, actual)
        
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)
        print(f"{test_id}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    
    print(f"\nAverage: P={sum(all_p)/len(all_p):.3f}, R={sum(all_r)/len(all_r):.3f}, F1={sum(all_f1)/len(all_f1):.3f}")

if __name__ == "__main__":
    main()