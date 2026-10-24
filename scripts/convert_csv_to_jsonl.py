import csv
import json
import os
import ast

def parse_array_field(val):
    """Safely parses string representations of lists like "['c1', 'c2']" or semi-colon separated items."""
    if not val or val.strip() == "":
        return []
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    # Fallback: split by semicolon if formatted as string
    return [x.strip() for x in val.split(";") if x.strip()]

def convert_csv_to_jsonl(csv_filepath: str, jsonl_filepath: str):
    os.makedirs(os.path.dirname(jsonl_filepath), exist_ok=True)
    
    samples = []
    with open(csv_filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Format rubric criteria cleanly into a readable text block for LLM prompts
            criteria_list = parse_array_field(row.get("criteria[]", ""))
            max_scores_list = parse_array_field(row.get("criteria_max_score[]", ""))
            
            rubric_formatted_lines = []
            if len(criteria_list) == len(max_scores_list) and len(criteria_list) > 0:
                for crit, max_s in zip(criteria_list, max_scores_list):
                    rubric_formatted_lines.append(f"- {crit} (Max Points: {max_s})")
                formatted_rubric = "\n".join(rubric_formatted_lines)
            else:
                formatted_rubric = "\n".join([f"- {c}" for c in criteria_list]) if criteria_list else row.get("criteria[]", "")

            sample = {
                "id": str(row["id"]),
                "question": str(row.get("question", "")),
                "desired_answer": str(row.get("desired_answer", "")),
                "student_answer": str(row.get("student_answer", "")),
                "rubric": formatted_rubric,
                "ground_truth_score": float(row["student_score"]) if row.get("student_score") else None,
                "metadata": {
                    "raw_criteria": criteria_list,
                    "criteria_scores": parse_array_field(row.get("criteria_score[]", "")),
                    "criteria_max_scores": max_scores_list
                }
            }
            samples.append(sample)

    with open(jsonl_filepath, mode="w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Successfully converted {len(samples)} rows from {csv_filepath} to {jsonl_filepath}")

if __name__ == "__main__":
    convert_csv_to_jsonl("data/training_data_moh_eng_rubrics.csv", "data/processed/data.jsonl")