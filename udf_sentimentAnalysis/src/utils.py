import os
import json
import re
from typing import Dict, Any
from dotenv import load_dotenv
from pathlib import Path

def get_project_root():
    """Returns the project root folder"""
    return Path(__file__).parent

def load_csv_content(database_id: str, gt_folder: str, table_cache: Dict[str, str]) -> str:
    """Load CSV file content as raw text."""
    if database_id in table_cache:
        return table_cache[database_id]
    
    csv_path = os.path.join(gt_folder, f"{database_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Table file not found: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_content = file.read()
        
        max_chars = 5000
        if len(csv_content) > max_chars:
            lines = csv_content.split('\n')
            header = lines[0]
            data_lines = lines[1:30]
            csv_content = header + '\n' + '\n'.join(data_lines)
            csv_content += f"\n\n(Note: Truncated to first 30 rows for processing)"
        
        table_cache[database_id] = csv_content
        return csv_content
        
    except Exception as e:
        raise Exception(f"Error loading table {database_id}: {str(e)}")
