def create_table_qa_prompt(question: str, csv_content: str) -> str:
    """Create a prompt for the LLM with raw CSV content."""
    prompt = f"""Analyze this CSV data and answer the question. Return ONLY the answer, nothing else.

QUESTION: {question}

CSV DATA:
{csv_content}

IMPORTANT INSTRUCTIONS:
- The first row contains column headers
- Analyze the CSV data directly to answer the question
- For sentiment analysis, use scores: -1.0 (very negative) to 1.0 (very positive)
- For questions asking for IDs/numbers, return a list like: [123, 456, 789]
- For questions asking for records/objects, return a list of objects with relevant fields
- For questions asking for text, return the text directly
- Do NOT include explanations, confidence scores, or metadata
- Do NOT wrap the answer in JSON structure unless the answer itself should be JSON
- Work directly with the CSV format - understand commas as separators and quotes as text delimiters

ANSWER:"""
    
    return prompt