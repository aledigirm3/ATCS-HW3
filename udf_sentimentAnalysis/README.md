# LLM Sentiment Analysis Benchmark

This repository contains resources to evaluate a Large Language Model (LLM) on sentiment analysis tasks, including tasks that extend beyond standard SQL capabilities. The structure and contents of the repository are organized as follows:

## Folder Structure

### `GT/` (Ground Truth)
- **`sentiment_analysis_example.json`**: A collection of natural language questions that require sentiment analysis, either standalone or in combination with operations beyond typical SQL functions. Relative expected results: Ground truth answers to be used as benchmarks when evaluating the LLM's responses.
- **Data Tables**: Supporting tables containing the data necessary to answer the questions.

### `src/` (Source Code)
- **`test_llm.py`**: Script to query a language model with the questions from `sentiment_analysis_example.json`. Supports both zero-shot and few-shot prompting strategies.
- **`evaluation.py`**: Script to evaluate the LLM's responses against the ground truth. Outputs evaluation results in JSON format, saved in a separate folder.

### `Results/`
- **`test_results.json`**: The LLM's generated answers to the benchmark questions.

## Purpose

The main goal of this project is to assess the capabilities of LLMs in handling complex sentiment analysis queries, particularly those that are difficult to resolve with conventional SQL-based approaches.

## Replicate the experiment

Install Python 3.11.11. Execute the following command.

```bash
git clone https://github.com/aledigirm3/ATCS-HW3.git
cd ATCS-HW3/udf_sentimentAnalisys
pip install -r requirements.txt
```

Before executing the scripts, you must create a .env file in the root directory of the project. Use the structure provided in the .env.example file, replacing 'GROQ_API_KEY' with your personal key obtained from Groq.
```env
# Example .env file
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=model of choice
```

Navigate to the 'src' directory:

```bash
  cd src
```

Now run these script

```bash
  python test_llm.py
```

This script generates a response for each of the questions contained in the sentiment_analysis_example.json over various entities stored in the tables within the GT folder by utilizing a Large Language Model (LLM) and saves the output in a JSON file named 'test_results.json' in the Results folder.




## Evaluation

To perform the evaluation, simply execute the following script:

```bash
  python evaluation.py
```
Performance metrics are precision, recall and f1 score for the first 9 examples or computed using 'BERTScore' for the last example because it includes summarization, and printed to the shell.
