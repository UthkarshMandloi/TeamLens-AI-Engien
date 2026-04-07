# -----------------------------------------------------------------------------
# RUN THIS IN GOOGLE COLAB (Select T4 GPU in Runtime -> Change runtime type)
# -----------------------------------------------------------------------------
# First, install the required libraries in your Colab cell:
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# -----------------------------------------------------------------------------

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
max_seq_length = 2048
dtype = None # None for auto-detection
load_in_4bit = True # Use 4bit quantization to save massive amounts of memory

print("Loading Base LLaMA-3 Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # Pre-quantized for speed
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Add LoRA Adapters (This makes the model train fast by only updating 1-10% of weights)
print("Applying LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 3. Load Your Custom Dataset
print("Loading Custom Dataset...")
# IMPORTANT: Upload your task_extraction_dataset.jsonl to Colab before running!
dataset = load_dataset("json", data_files="task_extraction_dataset.jsonl", split="train")

def formatting_prompts_func(examples):
    # This function formats your JSONL data into the specific chat format LLaMA-3 expects
    texts = []
    for messages in examples["messages"]:
        # We use the tokenizer to apply the LLaMA-3 chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Setup The Trainer
print("Setting up Trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Increase this to ~100-200 for better results on larger datasets
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. Execute Training!
print("🚀 Starting Fine-Tuning Process...")
trainer_stats = trainer.train()
print("✅ Training Complete!")

# 6. Save the Custom Model
print("Saving TeamLens AI Model...")
model.save_pretrained("teamlens_llama3_lora")
tokenizer.save_pretrained("teamlens_llama3_lora")

print("🎉 DONE! You now have a custom fine-tuned LLaMA-3 model.")