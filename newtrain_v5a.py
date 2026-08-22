#!/usr/bin/env python3
"""
Compatifi V5A - Full Dataset Continual-Learning Training

Uses ALL valid samples from:
    V5A main dataset
    V4A replay dataset
    V4B replay dataset

Important:
- V4A/V4B instruction/input/output are preserved.
- V4A/V4B are NOT converted into V5A labels.
- Each task gets its own system prompt.
- The instruction determines which task the model performs.
- The same model learns all three tasks.
- Replay reduces catastrophic forgetting.

Expected project layout:
    ~/V5/
      datasets/
        v5a/
          v5a_main.jsonl
          v4a_replay.jsonl
          v4b_replay.jsonl
      newtrain_v5a_full.py
"""

import os
import sys
import json
import random
import shutil
import inspect
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    trainer_utils,
)
from trl import SFTConfig, SFTTrainer


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Base model: your V4B merged model.
MODEL_NAME = "../V4B_Final_Merged_Model"

# Dataset paths.
V5A_DATASET_PATH = "./datasets/v5a/v5a_main.jsonl"
V4A_REPLAY_PATH = "./datasets/v5a/v4a_replay.jsonl"
V4B_REPLAY_PATH = "./datasets/v5a/v4b_replay.jsonl"

# IMPORTANT:
# New output directory so this run does NOT accidentally resume the old run.
OUTPUT_DIR = "./v5a_full_checkpoints"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")

# Rebuild formatted dataset every run.
CACHE_DIR = "./dataset_cache_v5a_full"

# Use every valid record.
USE_FULL_DATASET = True

# Training.
MAX_SEQ_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 2

# Checkpoint / logging.
SAVE_STEPS = 1000
LOGGING_STEPS = 25
SAVE_TOTAL_LIMIT = 3

# Validation.
VALIDATION_RATIO = 0.05

# LoRA.
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

# Reproducibility.
SEED = 42

# False = start a new run.
# True = resume the latest checkpoint in OUTPUT_DIR.
RESUME_FROM_CHECKPOINT = False

# Rebuild cache.
REBUILD_CACHE = True

# If malformed lines exist:
# True  -> report them and train using valid records.
# False -> stop immediately.
ALLOW_VALID_RECORDS_WITH_BAD_LINES = True


# ==============================================================================
# 2. TASK DEFINITIONS
# ==============================================================================

V5A_INSTRUCTION = "Predict the assistant objective"
V4A_INSTRUCTION = "Extract long-term memories"
V4B_INSTRUCTION = "Extract long-term people memories"

TASK_NAMES = ("V5A", "V4A", "V4B")

EXPECTED_V5A_FIELDS = {
    "primary_objective",
    "secondary_objective",
    "priority",
    "reason",
}

VALID_PRIMARY_OBJECTIVES = {
    "Emotional Support",
    "Reduce Anxiety",
    "Solve Problem",
    "Decision Support",
    "Planning Assistance",
    "Motivation",
    "Information Sharing",
    "Maintain Rapport",
}

VALID_SECONDARY_OBJECTIVES = {
    "None",
    "Build Confidence",
    "Increase Confidence",
    "Clarify Situation",
    "Encourage Reflection",
    "Suggest Next Steps",
    "Maintain Rapport",
}

VALID_PRIORITIES = {
    "High",
    "Medium",
    "Low",
}


# ==============================================================================
# 3. TASK-SPECIFIC SYSTEM PROMPTS
# ==============================================================================

# CRITICAL:
# Do NOT use the V5A system prompt for V4A/V4B replay.
#
# The previous problem was:
#
#   System: Do not extract memories.
#   User:   Extract long-term memories.
#   Target: V4A memory output.
#
# That creates contradictory supervision.
#
# This version gives every task its own system prompt.

V5A_SYSTEM_PROMPT = """
You are Compatifi V5A.

Your ONLY task is to predict the immediate objective that the assistant
should accomplish in the current conversation.

Do not generate an assistant reply.
Do not extract long-term memories.
Do not extract people memories.
Do not generate personality information.
Do not generate a conversation summary.
Do not predict future events.

Return EXACTLY ONE JSON object containing EXACTLY these four keys:

{
  "primary_objective": "...",
  "secondary_objective": "...",
  "priority": "...",
  "reason": "..."
}

Valid primary objectives:
- Emotional Support
- Reduce Anxiety
- Solve Problem
- Decision Support
- Planning Assistance
- Motivation
- Information Sharing
- Maintain Rapport

Valid secondary objectives:
- None
- Build Confidence
- Increase Confidence
- Clarify Situation
- Encourage Reflection
- Suggest Next Steps
- Maintain Rapport

Valid priorities:
- High
- Medium
- Low

Rules:
1. Always select ONE primary objective.
2. The secondary objective supports the primary objective.
3. Use "None" when no meaningful secondary objective is supported.
4. The reason must use only information explicitly present in the conversation.
5. Output EXACTLY the four required keys.
6. Never output memories, people, conversation, or other fields.
7. Return JSON only.
""".strip()


V4A_SYSTEM_PROMPT = """
You are Compatifi V4A.

Perform the original task specified by the instruction.

This is the original V4A long-term memory extraction task.
Preserve the original V4A output schema and semantics.

Do NOT convert this task into V5A assistant-objective prediction.
Do NOT add V5A fields.
Return the original V4A target structure.
""".strip()


V4B_SYSTEM_PROMPT = """
You are Compatifi V4B.

Perform the original task specified by the instruction.

This is the original V4B long-term people-memory extraction task.
Preserve the original V4B output schema and semantics.

Do NOT convert this task into V5A assistant-objective prediction.
Do NOT add V5A fields.
Return the original V4B target structure.
""".strip()


SYSTEM_PROMPTS = {
    "V5A": V5A_SYSTEM_PROMPT,
    "V4A": V4A_SYSTEM_PROMPT,
    "V4B": V4B_SYSTEM_PROMPT,
}


# ==============================================================================
# 4. REPRODUCIBILITY / GPU
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if not torch.cuda.is_available():
    print("=" * 80)
    print("ERROR: CUDA IS NOT AVAILABLE")
    print("=" * 80)
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0)

if torch.cuda.is_bf16_supported():
    COMPUTE_DTYPE = torch.bfloat16
    USE_BF16 = True
    USE_FP16 = False
else:
    COMPUTE_DTYPE = torch.float16
    USE_BF16 = False
    USE_FP16 = True

print("=" * 80)
print("GPU / PRECISION")
print("=" * 80)
print(f"GPU           : {GPU_NAME}")
print(f"Compute dtype : {COMPUTE_DTYPE}")
print(f"BF16          : {USE_BF16}")
print(f"FP16          : {USE_FP16}")
print("=" * 80)


# ==============================================================================
# 5. BASIC HELPERS
# ==============================================================================

def is_empty_value(value):
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0

    return False


def normalize_json_value(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def record_fingerprint(record):
    # Ignore internal task metadata when checking duplicates.
    clean = {
        k: v
        for k, v in record.items()
        if not k.startswith("_")
    }

    return normalize_json_value(clean)


def instruction_to_task(instruction):
    instruction = instruction.strip()

    if instruction == V5A_INSTRUCTION:
        return "V5A"

    if instruction == V4A_INSTRUCTION:
        return "V4A"

    if instruction == V4B_INSTRUCTION:
        return "V4B"

    return "UNKNOWN"


# ==============================================================================
# 6. RECORD VALIDATION
# ==============================================================================

def validate_common_record(item, path, line_number):
    if not isinstance(item, dict):
        raise ValueError("Record is not a JSON object.")

    required = ("instruction", "input", "output")

    for field in required:
        if field not in item:
            raise ValueError(
                f"Missing required field: {field}"
            )

    if not isinstance(item["instruction"], str):
        raise ValueError("instruction must be a string.")

    if not item["instruction"].strip():
        raise ValueError("instruction is empty.")

    if is_empty_value(item["input"]):
        raise ValueError("input is completely empty.")

    if is_empty_value(item["output"]):
        raise ValueError("output is completely empty.")


def validate_v5a_output(item, index):
    output = item["output"]

    if not isinstance(output, dict):
        raise ValueError(
            f"V5A record {index}: output must be a JSON object."
        )

    fields = set(output.keys())

    if fields != EXPECTED_V5A_FIELDS:
        extra = fields - EXPECTED_V5A_FIELDS
        missing = EXPECTED_V5A_FIELDS - fields

        raise ValueError(
            f"V5A schema error at record {index}. "
            f"Extra={sorted(extra)}, Missing={sorted(missing)}"
        )

    for field in EXPECTED_V5A_FIELDS:
        if is_empty_value(output[field]):
            raise ValueError(
                f"V5A record {index}: {field!r} is empty."
            )

    # Warn instead of changing labels.
    if output["primary_objective"] not in VALID_PRIMARY_OBJECTIVES:
        print(
            f"WARNING: unknown V5A primary_objective at "
            f"record {index}: {output['primary_objective']!r}"
        )

    if output["secondary_objective"] not in VALID_SECONDARY_OBJECTIVES:
        print(
            f"WARNING: unknown V5A secondary_objective at "
            f"record {index}: {output['secondary_objective']!r}"
        )

    if output["priority"] not in VALID_PRIORITIES:
        print(
            f"WARNING: unknown V5A priority at "
            f"record {index}: {output['priority']!r}"
        )


# ==============================================================================
# 7. JSONL LOADING
# ==============================================================================

def load_jsonl(path, expected_task):
    path = Path(path)

    print()
    print("=" * 80)
    print(f"VALIDATING: {path}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: Required dataset file does not exist: {path}")
        sys.exit(1)

    records = []
    invalid_count = 0
    duplicate_count = 0

    fingerprints = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, 1):

            if not raw_line.strip():
                continue

            try:
                item = json.loads(raw_line)

                validate_common_record(
                    item,
                    path,
                    line_number,
                )

                task = instruction_to_task(
                    item["instruction"]
                )

                if task != expected_task:
                    raise ValueError(
                        f"Expected {expected_task}, "
                        f"but instruction maps to {task!r}."
                    )

                # V5A has a strict schema.
                # V4A/V4B intentionally do not have to match it.
                if task == "V5A":
                    validate_v5a_output(
                        item,
                        len(records),
                    )

                fingerprint = record_fingerprint(item)

                if fingerprint in fingerprints:
                    duplicate_count += 1
                    print(
                        f"[DUPLICATE] {path.name} line {line_number}"
                    )
                else:
                    fingerprints.add(fingerprint)

                item["_task"] = task

                records.append(item)

            except Exception as exc:
                invalid_count += 1

                print(
                    f"[INVALID] {path.name} line {line_number}: {exc}"
                )

    print()
    print(f"Valid records   : {len(records):,}")
    print(f"Invalid records : {invalid_count:,}")
    print(f"Duplicate records: {duplicate_count:,}")

    if invalid_count and not ALLOW_VALID_RECORDS_WITH_BAD_LINES:
        print()
        print("ERROR: Invalid records found. Stopping.")
        sys.exit(1)

    if not records:
        print("ERROR: No valid records found.")
        sys.exit(1)

    return (
        Dataset.from_list(records),
        {
            "valid": len(records),
            "invalid": invalid_count,
            "duplicates": duplicate_count,
        },
    )


# ==============================================================================
# 8. LOAD ALL THREE DATASETS
# ==============================================================================

# Main V5A is mandatory.
v5a_dataset, v5a_stats = load_jsonl(
    V5A_DATASET_PATH,
    "V5A",
)

# V4A replay.
v4a_dataset, v4a_stats = load_jsonl(
    V4A_REPLAY_PATH,
    "V4A",
)

# V4B replay.
v4b_dataset, v4b_stats = load_jsonl(
    V4B_REPLAY_PATH,
    "V4B",
)


# ==============================================================================
# 9. USE THE FULL DATASET
# ==============================================================================

# IMPORTANT:
# There is NO .select() here.
# Every valid record is retained.

v5a_dataset = v5a_dataset.shuffle(
    seed=SEED
)

v4a_dataset = v4a_dataset.shuffle(
    seed=SEED + 1
)

v4b_dataset = v4b_dataset.shuffle(
    seed=SEED + 2
)

print()
print("=" * 80)
print("FULL DATASET MODE")
print("=" * 80)
print(f"V5A records : {len(v5a_dataset):,}")
print(f"V4A records : {len(v4a_dataset):,}")
print(f"V4B records : {len(v4b_dataset):,}")

total_available = (
    len(v5a_dataset)
    + len(v4a_dataset)
    + len(v4b_dataset)
)

print(f"TOTAL       : {total_available:,}")


# ==============================================================================
# 10. TASK IDENTITY SAFETY CHECK
# ==============================================================================

def assert_task_identity(dataset, expected_task):
    for index, example in enumerate(dataset):
        actual_task = instruction_to_task(
            example["instruction"]
        )

        if actual_task != expected_task:
            raise RuntimeError(
                f"Task identity mismatch at index {index}: "
                f"expected {expected_task}, got {actual_task}"
            )

        if example.get("_task") != expected_task:
            raise RuntimeError(
                f"Internal task marker mismatch at index {index}."
            )


assert_task_identity(v5a_dataset, "V5A")
assert_task_identity(v4a_dataset, "V4A")
assert_task_identity(v4b_dataset, "V4B")


# ==============================================================================
# 11. COMBINE ALL DATA
# ==============================================================================

combined_dataset = concatenate_datasets(
    [
        v5a_dataset,
        v4a_dataset,
        v4b_dataset,
    ]
)

combined_dataset = combined_dataset.shuffle(
    seed=SEED
)

combined_counts = Counter(
    combined_dataset["_task"]
)

combined_total = len(combined_dataset)

print()
print("=" * 80)
print("FULL COMBINED DATASET DISTRIBUTION")
print("=" * 80)

for task in TASK_NAMES:
    count = combined_counts.get(task, 0)
    pct = (
        count / combined_total * 100
        if combined_total
        else 0
    )

    print(
        f"{task}: {count:,} ({pct:.2f}%)"
    )

replay_count = (
    combined_counts.get("V4A", 0)
    + combined_counts.get("V4B", 0)
)

replay_pct = (
    replay_count / combined_total * 100
    if combined_total
    else 0
)

print(
    f"Replay total: {replay_count:,} "
    f"({replay_pct:.2f}%)"
)

if replay_pct > 15:
    print("WARNING: Replay is above 15%.")

if replay_pct < 2:
    print("WARNING: Replay is below 2%.")


# ==============================================================================
# 12. LOAD TOKENIZER BEFORE FORMAT_DATASET
# ==============================================================================

print()
print("=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Tokenizer loaded.")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


# ==============================================================================
# 13. FORMAT DATASET
# ==============================================================================

def format_sample(example):
    """
    Create instruction-following training text.

    The original task instruction/input/output remain intact.

    Only the system prompt is task-specific:
        V5A -> V5A system prompt
        V4A -> V4A system prompt
        V4B -> V4B system prompt
    """

    task = instruction_to_task(
        example["instruction"]
    )

    if task not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown task instruction: "
            f"{example['instruction']!r}"
        )

    instruction = example["instruction"].strip()

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

    # IMPORTANT:
    # No "Output:" label is added to the user prompt.
    # The assistant message itself is the target.
    user_content = (
        f"Task: {instruction}\n\n"
        f"Input:\n{input_data}"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPTS[task],
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": output_data,
        },
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": formatted_text,
        "task": task,
    }


# ==============================================================================
# 14. REBUILD CACHE
# ==============================================================================

if REBUILD_CACHE and os.path.exists(CACHE_DIR):
    print()
    print("=" * 80)
    print("REMOVING OLD CACHE")
    print("=" * 80)

    shutil.rmtree(CACHE_DIR)


print()
print("=" * 80)
print("FORMATTING FULL DATASET")
print("=" * 80)

formatted_dataset = combined_dataset.map(
    format_sample,
    remove_columns=combined_dataset.column_names,
    desc="Formatting V5A + V4A + V4B",
)

formatted_dataset.save_to_disk(
    CACHE_DIR
)

print(
    f"Formatted cache saved to: {CACHE_DIR}"
)


# ==============================================================================
# 15. TOKEN LENGTH STATISTICS
# ==============================================================================

print()
print("=" * 80)
print("TOKEN LENGTH STATISTICS")
print("=" * 80)

for task in TASK_NAMES:

    task_indices = [
        i
        for i, task_name in enumerate(
            formatted_dataset["task"]
        )
        if task_name == task
    ]

    if not task_indices:
        continue

    sample_count = min(
        1000,
        len(task_indices),
    )

    rng = random.Random(
        SEED + TASK_NAMES.index(task)
    )

    selected_indices = rng.sample(
        task_indices,
        sample_count,
    )

    lengths = []

    for i in selected_indices:
        token_ids = tokenizer.encode(
            formatted_dataset[i]["text"],
            add_special_tokens=True,
        )

        lengths.append(
            len(token_ids)
        )

    lengths.sort()

    median = lengths[
        len(lengths) // 2
    ]

    p95 = lengths[
        min(
            len(lengths) - 1,
            int(len(lengths) * 0.95),
        )
    ]

    over_limit = sum(
        x > MAX_SEQ_LENGTH
        for x in lengths
    )

    print()
    print(task)
    print(
        f"  checked       : {sample_count}"
    )
    print(
        f"  average       : "
        f"{sum(lengths) / len(lengths):.2f}"
    )
    print(
        f"  minimum       : {min(lengths)}"
    )
    print(
        f"  median        : {median}"
    )
    print(
        f"  p95           : {p95}"
    )
    print(
        f"  maximum       : {max(lengths)}"
    )
    print(
        f"  > max length  : "
        f"{over_limit}/{len(lengths)}"
    )


# ==============================================================================
# 16. STRATIFIED TRAIN / VALIDATION SPLIT
# ==============================================================================

def stratified_split(
    dataset,
    validation_ratio,
    seed,
):
    """
    Preserve V5A/V4A/V4B proportions separately in validation.

    Every task gets its own split, then the three train pieces and three
    validation pieces are recombined.
    """

    rng = random.Random(seed)

    train_parts = []
    validation_parts = []

    for task in TASK_NAMES:

        indices = [
            i
            for i, task_name in enumerate(
                dataset["task"]
            )
            if task_name == task
        ]

        rng.shuffle(indices)

        if len(indices) <= 1:
            validation_count = 0
        else:
            validation_count = max(
                1,
                int(
                    round(
                        len(indices)
                        * validation_ratio
                    )
                ),
            )

            validation_count = min(
                validation_count,
                len(indices) - 1,
            )

        validation_indices = indices[
            :validation_count
        ]

        train_indices = indices[
            validation_count:
        ]

        train_parts.append(
            dataset.select(
                train_indices
            )
        )

        if validation_indices:
            validation_parts.append(
                dataset.select(
                    validation_indices
                )
            )

    train_dataset = concatenate_datasets(
        train_parts
    ).shuffle(seed=seed)

    eval_dataset = concatenate_datasets(
        validation_parts
    ).shuffle(seed=seed + 1)

    return train_dataset, eval_dataset


train_dataset, eval_dataset = stratified_split(
    formatted_dataset,
    VALIDATION_RATIO,
    SEED,
)


# ==============================================================================
# 17. PRINT SPLIT DISTRIBUTIONS
# ==============================================================================

def print_distribution(dataset, title):
    counts = Counter(
        dataset["task"]
    )

    total = len(dataset)

    print()
    print(title)

    for task in TASK_NAMES:

        count = counts.get(task, 0)

        pct = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"  {task}: "
            f"{count:,} ({pct:.2f}%)"
        )


print()
print("=" * 80)
print("TRAIN / VALIDATION")
print("=" * 80)

print(
    f"Train      : {len(train_dataset):,}"
)

print(
    f"Validation : {len(eval_dataset):,}"
)

print_distribution(
    train_dataset,
    "TRAIN DISTRIBUTION",
)

print_distribution(
    eval_dataset,
    "VALIDATION DISTRIBUTION",
)


# ==============================================================================
# 18. LOAD 4-BIT MODEL
# ==============================================================================

print()
print("=" * 80)
print("LOADING V4B BASE MODEL")
print("=" * 80)

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
    trust_remote_code=True,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model
)

print("Base model loaded and prepared for LoRA.")


# ==============================================================================
# 19. SFT CONFIG
# ==============================================================================

sft_parameters = inspect.signature(
    SFTConfig.__init__
).parameters

training_kwargs = {
    "output_dir": OUTPUT_DIR,

    "per_device_train_batch_size":
        PER_DEVICE_BATCH_SIZE,

    "per_device_eval_batch_size":
        PER_DEVICE_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "learning_rate":
        LEARNING_RATE,

    "num_train_epochs":
        NUM_TRAIN_EPOCHS,

    "warmup_ratio":
        0.03,

    "weight_decay":
        0.01,

    "lr_scheduler_type":
        "cosine",

    "optim":
        "paged_adamw_8bit",

    "bf16":
        USE_BF16,

    "fp16":
        USE_FP16,

    "tf32":
        True,

    "gradient_checkpointing":
        True,

    "logging_steps":
        LOGGING_STEPS,

    "logging_first_step":
        True,

    "report_to":
        "none",

    "save_strategy":
        "steps",

    "save_steps":
        SAVE_STEPS,

    "save_total_limit":
        SAVE_TOTAL_LIMIT,

    "seed":
        SEED,

    "remove_unused_columns":
        True,

    "packing":
        False,
}


# Current TRL normally uses max_length.
if "max_length" in sft_parameters:

    training_kwargs[
        "max_length"
    ] = MAX_SEQ_LENGTH

elif "max_seq_length" in sft_parameters:

    training_kwargs[
        "max_seq_length"
    ] = MAX_SEQ_LENGTH

else:

    raise RuntimeError(
        "Your TRL version does not expose "
        "max_length or max_seq_length."
    )


# Current TRL uses eval_strategy.
# Older TRL may use evaluation_strategy.
if "eval_strategy" in sft_parameters:

    training_kwargs[
        "eval_strategy"
    ] = "steps"

    training_kwargs[
        "eval_steps"
    ] = SAVE_STEPS

elif "evaluation_strategy" in sft_parameters:

    training_kwargs[
        "evaluation_strategy"
    ] = "steps"

    training_kwargs[
        "eval_steps"
    ] = SAVE_STEPS


# Some versions accept dataset_text_field.
if "dataset_text_field" in sft_parameters:

    training_kwargs[
        "dataset_text_field"
    ] = "text"


training_args = SFTConfig(
    **training_kwargs
)


# ==============================================================================
# 20. LORA
# ==============================================================================

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)


# ==============================================================================
# 21. SFT TRAINER
# ==============================================================================

trainer_signature = inspect.signature(
    SFTTrainer.__init__
).parameters

trainer_kwargs = {
    "model":
        model,

    "train_dataset":
        train_dataset,

    "eval_dataset":
        eval_dataset,

    "args":
        training_args,

    "peft_config":
        lora_config,
}


if "processing_class" in trainer_signature:

    trainer_kwargs[
        "processing_class"
    ] = tokenizer

elif "tokenizer" in trainer_signature:

    trainer_kwargs[
        "tokenizer"
    ] = tokenizer

else:

    raise RuntimeError(
        "Your TRL SFTTrainer has neither "
        "processing_class nor tokenizer."
    )


trainer = SFTTrainer(
    **trainer_kwargs
)


# ==============================================================================
# 22. CHECKPOINT HANDLING
# ==============================================================================

last_checkpoint = None

if os.path.isdir(OUTPUT_DIR):

    last_checkpoint = (
        trainer_utils.get_last_checkpoint(
            OUTPUT_DIR
        )
    )

print()
print("=" * 80)
print("CHECKPOINT CONFIGURATION")
print("=" * 80)

if RESUME_FROM_CHECKPOINT:

    if last_checkpoint:

        print(
            f"Will resume from: "
            f"{last_checkpoint}"
        )

    else:

        print(
            "No checkpoint found."
        )

        print(
            "Starting fresh."
        )

else:

    print(
        "Fresh training requested."
    )

    print(
        "Old checkpoints will NOT be "
        "resumed automatically."
    )


# ==============================================================================
# 23. FINAL TRAINING SUMMARY
# ==============================================================================

print()
print("=" * 80)
print("COMPATIFI V5A FULL-DATASET TRAINING")
print("=" * 80)

print()
print("Base model:")
print(MODEL_NAME)

print()
print("Tasks:")
print("V5A = Predict the assistant objective")
print("V4A = Extract long-term memories")
print("V4B = Extract long-term people memories")

print()
print("FULL DATASET:")
print(
    f"V5A : {combined_counts['V5A']:,}"
)
print(
    f"V4A : {combined_counts['V4A']:,}"
)
print(
    f"V4B : {combined_counts['V4B']:,}"
)
print(
    f"TOTAL : {combined_total:,}"
)

print()
print("Training:")
print(
    f"Epochs: {NUM_TRAIN_EPOCHS}"
)
print(
    f"Batch: {PER_DEVICE_BATCH_SIZE}"
)
print(
    f"Gradient accumulation: "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)
print(
    f"Effective batch: "
    f"{PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)
print(
    f"Max sequence length: "
    f"{MAX_SEQ_LENGTH}"
)
print(
    f"Learning rate: "
    f"{LEARNING_RATE}"
)

print()
print("Replay:")
print(
    "ALL V4A and V4B replay records are retained."
)

print()
print(
    "Continual-learning objective:"
)
print(
    "Teach V5A while retaining V4A/V4B capabilities "
    "through replay."
)

print("=" * 80)


# ==============================================================================
# 24. TRAIN
# ==============================================================================

if RESUME_FROM_CHECKPOINT and last_checkpoint:

    train_result = trainer.train(
        resume_from_checkpoint=last_checkpoint
    )

else:

    train_result = trainer.train()


# ==============================================================================
# 25. SAVE FINAL MODEL
# ==============================================================================

print()
print("=" * 80)
print("SAVING FINAL V5A MODEL")
print("=" * 80)

os.makedirs(
    FINAL_MODEL_DIR,
    exist_ok=True
)

trainer.save_model(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)

print()
print("=" * 80)
print("V5A FULL-DATASET TRAINING COMPLETE")
print("=" * 80)

print(
    f"Final model: "
    f"{FINAL_MODEL_DIR}"
)

print("=" * 80)


# ==============================================================================
# 26. METRICS
# ==============================================================================

if hasattr(
    train_result,
    "metrics"
):

    print()
    print("=" * 80)
    print("FINAL TRAINING METRICS")
    print("=" * 80)

    for key, value in train_result.metrics.items():

        print(
            f"{key}: {value}"
        )

print()
print("=" * 80)
print("DONE")
print("=" * 80)
