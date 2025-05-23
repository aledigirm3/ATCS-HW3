import os
from typing import Dict, List, Any
from groq import Groq
from utils import load_csv_content
from prompt import create_table_qa_prompt



class UDFTableQA:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192", gt_folder: str = "GT"):
        """
        Initialize the UDF Table Question Answering system with Groq.
        
        Args:
            api_key: Groq API key
            model: Model to use (default: llama3-8b-8192)
            gt_folder: Folder containing CSV files (default: "GT")
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.gt_folder = gt_folder
        self._table_cache = {}
    
    def _load_csv_content(self, database_id: str) -> str:
        """Load CSV file content as raw text."""
        return load_csv_content(database_id, self.gt_folder, self._table_cache)
    
    def answer_table_question(self, question: str, database_id) -> Any:
        """
        Answer a question about table data that requires capabilities beyond standard SQL.
        
        Args:
            question: Natural language question
            database_id: Database identifier (string) or list of database identifiers (list of strings)
                        to load the corresponding CSV(s)
            
        Returns:
            Direct answer from the LLM
        """
        # Handle both string and list inputs
        if isinstance(database_id, str):
            csv_content = self._load_csv_content(database_id)
        elif isinstance(database_id, list):
            csv_contents = []
            for db_id in database_id:
                csv_contents.append(self._load_csv_content(db_id))
            # Combine all CSV contents
            csv_content = '\n\n'.join(csv_contents)
        else:
            raise ValueError("database_id must be either a string or a list of strings")
        
        prompt = create_table_qa_prompt(question, csv_content)
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data analyst working with CSV data. You must return ONLY the direct answer to the question, nothing else. No explanations, no metadata, no JSON structure unless the answer itself should be JSON - just the pure answer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error calling Groq API: {str(e)}")