import json
import random

# List of synthetic team chat examples
chat_samples = [
    {"chat": "Hey @divyansh, can you fix the navbar padding issue by tomorrow?", "assignee": "divyansh",
     "task": "fix the navbar padding issue", "deadline": "tomorrow"},
    {"chat": "/assign @love to setup the kafka consumer on sunday", "assignee": "love",
     "task": "setup the kafka consumer", "deadline": "sunday"},
    {"chat": "Uthkarsh will handle the XGBoost model training this weekend.", "assignee": "uthkarsh",
     "task": "handle the XGBoost model training", "deadline": "this weekend"},
    {"chat": "Can someone look into the database timeout errors?", "assignee": None, "task": None, "deadline": None},
    # Negative example
    {"chat": "I think we should use Next.js for this.", "assignee": None, "task": None, "deadline": None},
    # Negative example
    {"chat": "/assign @anushka research competitors by Friday", "assignee": "anushka", "task": "research competitors",
     "deadline": "Friday"},
]

system_prompt = """You are an AI Project Manager. Analyze the following chat message and extract any task assignments.
Return ONLY a valid JSON object with the keys: is_task (boolean), assignee (string or null), task_description (string or null), deadline (string or null)."""


def generate_jsonl_dataset(filename="task_extraction_dataset.jsonl", num_samples=100):
    """
    Generates a JSONL dataset formatted for LLaMA-3 fine-tuning (ChatML format).
    """
    print(f"Generating {num_samples} samples into {filename}...")

    with open(filename, 'w', encoding='utf-8') as f:
        for _ in range(num_samples):
            sample = random.choice(chat_samples)

            # Construct the expected JSON output from the model
            is_task = sample["task"] is not None
            expected_output = {
                "is_task": is_task,
                "assignee": sample["assignee"],
                "task_description": sample["task"],
                "deadline": sample["deadline"]
            }

            # Format required by LLaMA-3 / HuggingFace fine-tuning
            training_example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample["chat"]},
                    {"role": "assistant", "content": json.dumps(expected_output)}
                ]
            }

            # Write as a JSON Lines format (one JSON object per line)
            f.write(json.dumps(training_example) + '\n')

    print("✅ Dataset generation complete!")


if __name__ == "__main__":
    # Generate a sample dataset of 50 examples
    generate_jsonl_dataset(num_samples=50)