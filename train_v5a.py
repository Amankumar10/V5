# ==============================================================================
# COMPATIFI V5A - ASSISTANT OBJECTIVE MODEL (L40S OPTIMIZED)

# ==============================================================================
#
# Continual-learning / multi-task QLoRA training
#
# Main task:
#   V5A = Predict the assistant objective
#
# Replay tasks:
#   V4A = Extract long-term user memories
#   V4B = Extract long-term people memories
#
# Hardware target:
#   NVIDIA L40S (48GB VRAM)
#

# ==============================================================================

# ==============================================================================
# SECTION 1: IMPORTS

# ==============================================================================

import json
import os
import sys
import random

import numpy as np
import torch

from datasets import Dataset, concatenate_datasets, load_from_disk

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)

from trl import SFTConfig, SFTTrainer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    trainer_utils,
)

# ==============================================================================
# SECTION 2: HARDWARE SETTINGS

# ==============================================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if not torch.cuda.is_available():
    print("=" * 70)
    print("ERROR: CUDA is not available.")
    print("An NVIDIA GPU is required for this training script.")
    print("=" * 70)
    sys.exit(1)


GPU_NAME = torch.cuda.get_device_name(0)

print("=" * 70)
print("GPU DETECTED")
print("=" * 70)
print(GPU_NAME)
print("=" * 70)

# ==============================================================================
# SECTION 3: CONFIGURATION

# ==============================================================================

# ------------------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------------------

MODEL_NAME = "../V4B_Final_Merged_Model"


# ------------------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------------------

V5A_DATASET_PATH = "./datasets/v5a/v5a_main.jsonl"
V4A_REPLAY_PATH  = "./datasets/v5a/v4a_replay.jsonl"
V4B_REPLAY_PATH  = "./datasets/v5a/v4b_replay.jsonl"


# ------------------------------------------------------------------------------
# CACHE & OUTPUT
# ------------------------------------------------------------------------------

CACHE_DIR  = "./dataset_cache_v5a"
OUTPUT_DIR = "./v5a_checkpoints"


# ------------------------------------------------------------------------------
# DATASET SIZE
# ------------------------------------------------------------------------------

V5A_TARGET_SAMPLES = 59_000
V4A_REPLAY_SAMPLES = 4_000
V4B_REPLAY_SAMPLES = 4_000


# ------------------------------------------------------------------------------
# TRAINING (L40S 48GB OPTIMIZED)
# ------------------------------------------------------------------------------

MAX_SEQ_LENGTH = 1024

# 8 micro-batch * 2 accumulation steps = 16 effective batch size
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 2

SAVE_STEPS = 1000
LOGGING_STEPS = 25


# ------------------------------------------------------------------------------
# LoRA CONFIGURATION
# ------------------------------------------------------------------------------

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ------------------------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------------------------

SEED = 42

# ==============================================================================
# SECTION 4: REPRODUCIBILITY

# ==============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# SECTION 5: COMPUTE DTYPE

# ==============================================================================

if torch.cuda.is_bf16_supported():
    COMPUTE_DTYPE = torch.bfloat16
    USE_BF16 = True
    USE_FP16 = False
else:
    COMPUTE_DTYPE = torch.float16
    USE_BF16 = False
    USE_FP16 = True


print("=" * 70)
print("COMPUTE CONFIGURATION")
print("=" * 70)
print(f"Compute dtype : {COMPUTE_DTYPE}")
print(f"BF16          : {USE_BF16}")
print(f"FP16          : {USE_FP16}")
print("=" * 70)

# ==============================================================================
# SECTION 6: V5A MULTI-TASK SYSTEM PROMPT

# ==============================================================================

SYSTEM_PROMPT = """
You are Compatifi V5A.

Your primary task is to predict the assistant objective.

For the V5A task, identify:
- primary_objective
- secondary_objective
- priority
- reason

However, this model also uses replay data from earlier Compatifi models
to reduce catastrophic forgetting.

You may receive one of these tasks:
1. Predict the assistant objective
2. Extract long-term memories
3. Extract long-term people memories

IMPORTANT:
Always follow the task specified by the instruction.

If the instruction is:
"Predict the assistant objective"
perform the V5A assistant-objective prediction task.

If the instruction is:
"Extract long-term memories"
perform the V4A user-memory extraction task.

If the instruction is:
"Extract long-term people memories"
perform the V4B people-memory extraction task.

Do not change one task into another task.
Do not invent information.
Return only the requested structured output.
"""

# ==============================================================================
# SECTION 7: TOKENIZER

# ==============================================================================

print("=" * 70)
print(f"Loading tokenizer for {MODEL_NAME}...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ==============================================================================
# SECTION 8: MODEL - 4-BIT QLoRA

# ==============================================================================

print("=" * 70)
print(f"Loading model {MODEL_NAME}")
print("4-bit QLoRA + SDPA")
print("=" * 70)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
)

# Disabled gradient checkpointing for speed on L40S VRAM capacity
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# ==============================================================================
# SECTION 9: LoRA CONFIGURATION

# ==============================================================================

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)

# ==============================================================================
# SECTION 10: DATASET VALIDATION

# ==============================================================================

def validate_jsonl(path):
    print()
    print("=" * 70)
    print(f"VALIDATING: {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    valid = 0
    invalid = 0
    empty = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                empty += 1
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Record is not a JSON object")
                if "instruction" not in item:
                    raise ValueError("Missing 'instruction'")
                if "input" not in item:
                    raise ValueError("Missing 'input'")
                if "output" not in item:
                    raise ValueError("Missing 'output'")
                if not item["instruction"]:
                    raise ValueError("Empty instruction")
                valid += 1
            except Exception as e:
                invalid += 1
                print(f"[INVALID] {os.path.basename(path)} line {line_number}: {e}")

    print(f"Valid records   : {valid}")
    print(f"Invalid records : {invalid}")
    print(f"Empty lines     : {empty}")

    if valid == 0:
        print("ERROR: No valid records found.")
        sys.exit(1)

    return valid

# ==============================================================================
# SECTION 11: VALIDATE ALL DATASETS

# ==============================================================================

v5a_count = validate_jsonl(V5A_DATASET_PATH)
v4a_count = validate_jsonl(V4A_REPLAY_PATH)
v4b_count = validate_jsonl(V4B_REPLAY_PATH)

# ==============================================================================
# SECTION 12: LOAD DATASETS

# ==============================================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

v5a_dataset = Dataset.from_json(V5A_DATASET_PATH)
v4a_dataset = Dataset.from_json(V4A_REPLAY_PATH)
v4b_dataset = Dataset.from_json(V4B_REPLAY_PATH)

print(f"V5A available : {len(v5a_dataset)}")
print(f"V4A available : {len(v4a_dataset)}")
print(f"V4B available : {len(v4b_dataset)}")

# ==============================================================================
# SECTION 13: SHUFFLE + SELECT REPLAY

# ==============================================================================

v5a_dataset = v5a_dataset.shuffle(seed=SEED)
v4a_dataset = v4a_dataset.shuffle(seed=SEED)
v4b_dataset = v4b_dataset.shuffle(seed=SEED)

v5a_take = min(V5A_TARGET_SAMPLES, len(v5a_dataset))
v4a_take = min(V4A_REPLAY_SAMPLES, len(v4a_dataset))
v4b_take = min(V4B_REPLAY_SAMPLES, len(v4b_dataset))

v5a_dataset = v5a_dataset.select(range(v5a_take))
v4a_dataset = v4a_dataset.select(range(v4a_take))
v4b_dataset = v4b_dataset.select(range(v4b_take))

# ==============================================================================
# SECTION 14: ADD TASK LABEL

# ==============================================================================

def identify_task(example):
    instruction = example.get("instruction", "").strip()

    if instruction == "Predict the assistant objective":
        task = "V5A"
    elif instruction == "Extract long-term memories":
        task = "V4A"
    elif instruction == "Extract long-term people memories":
        task = "V4B"
    else:
        task = "UNKNOWN"

    example["task"] = task
    return example


v5a_dataset = v5a_dataset.map(identify_task)
v4a_dataset = v4a_dataset.map(identify_task)
v4b_dataset = v4b_dataset.map(identify_task)

# ==============================================================================
# SECTION 15: CHECK TASK IDENTITY

# ==============================================================================

def check_task(dataset, expected_task):
    for example in dataset:
        if example["task"] != expected_task:
            print()
            print("=" * 70)
            print("ERROR: TASK IDENTITY MISMATCH")
            print("=" * 70)
            print(f"Expected   : {expected_task}")
            print(f"Found      : {example['task']}")
            print(f"Instruction: {example['instruction']}")
            sys.exit(1)


check_task(v5a_dataset, "V5A")
check_task(v4a_dataset, "V4A")
check_task(v4b_dataset, "V4B")

# ==============================================================================
# SECTION 16: COMBINE DATASETS

# ==============================================================================

print()
print("=" * 70)
print("COMBINING V5A + REPLAY")
print("=" * 70)

combined_dataset = concatenate_datasets([v5a_dataset, v4a_dataset, v4b_dataset])
combined_dataset = combined_dataset.shuffle(seed=SEED)

print(f"V5A new data : {len(v5a_dataset)}")
print(f"V4A replay   : {len(v4a_dataset)}")
print(f"V4B replay   : {len(v4b_dataset)}")
print("-" * 70)
print(f"TOTAL        : {len(combined_dataset)}")

total = len(combined_dataset)

print()
print("DATASET MIX")
print(f"V5A: {len(v5a_dataset) / total * 100:.2f}%")
print(f"V4A: {len(v4a_dataset) / total * 100:.2f}%")
print(f"V4B: {len(v4b_dataset) / total * 100:.2f}%")

replay_percentage = ((len(v4a_dataset) + len(v4b_dataset)) / total * 100)
print(f"Replay total: {replay_percentage:.2f}%")

if replay_percentage > 15:
    print("\nWARNING: Replay exceeds 15%.")
if replay_percentage < 2:
    print("\nWARNING: Replay is below 2%.")

# ==============================================================================
# SECTION 17: FORMAT SAMPLE

# ==============================================================================

def format_sample(example):
    instruction = example["instruction"]

    input_data = json.dumps(
        example["input"],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    output_data = json.dumps(
        example["output"],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    user_content = f"Task: {instruction}\n\nInput:\n{input_data}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_data},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": formatted_text}

# ==============================================================================
# SECTION 18: CACHE FORMATTED DATASET

# ==============================================================================

if os.path.exists(CACHE_DIR):
    print()
    print("=" * 70)
    print(f"Loading formatted dataset from cache: {CACHE_DIR}")
    print("=" * 70)
    formatted_dataset = load_from_disk(CACHE_DIR)
else:
    print()
    print("=" * 70)
    print("Formatting dataset with chat template")
    print("=" * 70)
    formatted_dataset = combined_dataset.map(
        format_sample,
        remove_columns=combined_dataset.column_names,
        desc="Formatting V5A + V4A + V4B",
    )
    print()
    print(f"Caching formatted dataset to: {CACHE_DIR}")
    formatted_dataset.save_to_disk(CACHE_DIR)

# ==============================================================================
# SECTION 19: DATASET STATISTICS

# ==============================================================================

stats_file = os.path.join(CACHE_DIR, "stats.json")

if os.path.exists(stats_file):
    print("Loading dataset statistics from cache...")
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)
else:
    print("Computing dataset statistics...")
    sample_size = min(1000, len(formatted_dataset))
    sample = formatted_dataset.select(range(sample_size))

    token_lengths = [
        len(tokenizer.encode(item["text"], add_special_tokens=True))
        for item in sample
    ]

    stats = {
        "dataset_size": len(formatted_dataset),
        "sample_size": sample_size,
        "avg_length": sum(token_lengths) / len(token_lengths),
        "max_length": max(token_lengths),
        "min_length": min(token_lengths),
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


print()
print("=" * 70)
print("DATASET STATISTICS")
print("=" * 70)
print(f"Dataset Size         : {stats['dataset_size']}")
print(f"Sample Size          : {stats['sample_size']}")
print(f"Average Token Length : {stats['avg_length']:.2f}")
print(f"Maximum Token Length : {stats['max_length']}")
print(f"Minimum Token Length : {stats['min_length']}")
print("=" * 70)

# ==============================================================================
# SECTION 20: TRAIN / VALIDATION SPLIT

# ==============================================================================

print()
print("=" * 70)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 70)

split_dataset = formatted_dataset.train_test_split(test_size=0.05, seed=SEED)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(eval_dataset)}")

# ==============================================================================
# SECTION 21: TRAINING CONFIGURATION

# ==============================================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Batch Configuration (Optimized for L40S 48GB)
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # Sequence Configuration
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,

    # Optimization
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",

    # Compute Precision
    bf16=USE_BF16,
    fp16=USE_FP16,
    tf32=True,

    # Speed Optimization: Gradient Checkpointing disabled for L40S VRAM headroom
    gradient_checkpointing=False,

    # Logging and Tracking
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    report_to="none",

    # Evaluation and Saving Strategy
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    disable_tqdm=False,
    seed=SEED,
    remove_unused_columns=True,
)

# ==============================================================================
# SECTION 22: TRAINER

# ==============================================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# ==============================================================================
# SECTION 23: CHECKPOINT RESUME

# ==============================================================================

print()
print("=" * 70)
print("CHECKING FOR EXISTING CHECKPOINTS")
print("=" * 70)

last_checkpoint = None

if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = trainer_utils.get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint:
    print(f"Resuming from checkpoint:\n{last_checkpoint}")
else:
    print("No checkpoint found.\nStarting a fresh V5A training run.")

# ==============================================================================
# SECTION 24: TRAIN

# ==============================================================================

print()
print("=" * 70)
print("STARTING COMPATIFI V5A TRAINING")
print("=" * 70)

print("\nMain task:")
print("V5A - Predict the assistant objective")

print("\nReplay tasks:")
print("V4A - Extract long-term memories")
print("V4B - Extract long-term people memories")

print(f"\nV5A samples : {len(v5a_dataset)}")
print(f"V4A replay  : {len(v4a_dataset)}")
print(f"V4B replay  : {len(v4b_dataset)}")
print(f"Total       : {len(combined_dataset)}")

print("\n" + "=" * 70)

trainer.train(resume_from_checkpoint=last_checkpoint)

# ==============================================================================
# SECTION 25: SAVE FINAL MODEL

# ==============================================================================

FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")

print()
print("=" * 70)
print("SAVING FINAL V5A MODEL")
print("=" * 70)

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print()
print("=" * 70)
print("V5A TRAINING COMPLETE")
print("=" * 70)
print(f"Final model:\n{FINAL_MODEL_DIR}")
print("=" * 70)

# trainer.train(
#     resume_from_checkpoint="./v5a_checkpoints/checkpoint-7956"
# )