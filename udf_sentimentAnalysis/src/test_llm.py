import os
from dotenv import load_dotenv
import json
from tableQA import UDFTableQA
from pathlib import Path
from utils import get_project_root





def main():
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    model_name=os.getenv('MODEL_NAME')
    if not groq_api_key:
        print("Please set GROQ_API_KEY environment variable")
        return
    
    data_dir = get_project_root().parent / 'GT'
    qa_system = UDFTableQA(api_key=groq_api_key,model=model_name,gt_folder=data_dir)

    test_results=[]
    
    try:
        json_path=data_dir / 'sentiment_analysis_examples.json'
        with open(json_path, 'r') as f:
            examples = json.load(f)
        
        print("=== Testing All Example ===")
        for example in examples:
            result = qa_system.answer_table_question(
                question=example['question'],
                database_id=example['database_id']
            )
            test_results.append({
            "test_id": example['unique_id'],
            "question": example['question'],
            "expected": example['expected_result'],
            "actual": result,
            })

        json_results_path = get_project_root().parent/ 'Results' / 'test_results_DeepSeek.json'
        # Save results
        with open(json_results_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()