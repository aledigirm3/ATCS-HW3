import os
import pandas as pd
import glob
from llm import query_groq

def summarize(entity: str, descriptions: list[str]) -> str:

    """
    Make summarization of entity descriptions.
    
    Args:
        query (list[str]): Natural language query.
        
    Returns:
        str: relevant tables.
    """

    system_prompt = """You are a helpful assistant skilled in summarizing subjective descriptions provided by multiple individuals. 

Your task is to read a numbered list of brief descriptions, each expressing a personal impression or opinion about a single, unnamed entity. The entity may be anything — such as a product, a book, a film, a public figure, or something else entirely — and is intentionally not specified. 

Based solely on the content of the descriptions, identify common themes, sentiments, and relevant insights. Then, write a coherent, concise summary that fairly represents the overall perspectives given, while acknowledging any diversity of opinion when present.

Do not assume any specific type of entity. Infer meaning purely from the input. The summary should be neutral, informative, and human-like in tone."""

    content = "\n".join(f"{i+1} {desc}" for i, desc in enumerate(descriptions))
    content = f"""Please summarize the following personal descriptions:
    
[ENTITY]:
{entity}

[DESCRIPTIONS]:
{content}"""


    return query_groq(messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ])


folder = '../GT/*.csv'

# All tables
files = glob.glob(folder)
with open('../llmResponse.txt', "a", encoding="utf-8") as file:
    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0]
        
        df = pd.read_csv(file)
        
        entity = df['entity_name'][0]
        descriptions = df['description'].tolist()

        summarization = summarize(entity, descriptions)
        
        file.write(f"Table: {filename}\n")
        file.write(f"llmRESPONSE: {'# SUMMARIZATION #'}\n\n")
        file.flush()
    
