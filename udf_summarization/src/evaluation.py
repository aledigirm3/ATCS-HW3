from bert_score import score
import json
from ansi_colors import *
from transformers import logging
logging.set_verbosity_error()

with open("../llmResponse.json", "r", encoding="utf-8") as f:
    llmResponses = json.load(f)
with open("../GT/summarization_examples.json", "r", encoding="utf-8") as f:
    gt = json.load(f)


precision_list = []
recall_list = []
f1_list = []

for llm_item, gt_item in zip(llmResponses, gt):

    llm_summary = llm_item['summary']
    gt_summary = gt_item['expected_result']

    P, R, F1 = score([gt_summary], [llm_summary], lang="en", verbose=False)

    precision_list.append(P.item())
    recall_list.append(R.item())
    f1_list.append(F1.item())
    
avg_precision = sum(precision_list) / len(precision_list)
avg_recall = sum(recall_list) / len(recall_list)
avg_f1 = sum(f1_list) / len(f1_list)

print(f"\n--- {CYAN}PERFORMANCE METRICS{RESET} ---")
print(f"avg PRECISION: {GREEN}{avg_precision}{RESET}")
print(f"avg RECALL: {GREEN}{avg_recall}{RESET}")
print(f"avg F1-score: {GREEN}{avg_f1}{RESET}\n")


