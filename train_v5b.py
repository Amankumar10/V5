#!/usr/bin/env python3
"""
Compatifi V5B full-dataset continual-learning trainer.

Main task:
    V5B = Predict assistant communication strategy

Replay:
    V5A = Predict the assistant objective
    V4A = Extract long-term memories
    V4B = Extract long-term people memories

All valid records are used. No dataset sampling.
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
# CONFIGURATION
# ==============================================================================

# V5B continues from the V5A merged model.
MODEL_NAME = "../V5A_Final_Merged_Model"

V5B_DATASET_PATH = "./datasets/v5b/v5b_main.jsonl"
V5A_REPLAY_PATH = "./datasets/replay/v5a_replay.jsonl"
V4A_REPLAY_PATH = "./datasets/replay/v4a_replay.jsonl"
V4B_REPLAY_PATH = "./datasets/replay/v4b_replay.jsonl"

OUTPUT_DIR = "./v5b_full_checkpoints"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
CACHE_DIR = "./dataset_cache_v5b_full"

# Full dataset mode: every valid record is retained.
USE_FULL_DATASET = True

# V5B specification.
MAX_SEQ_LENGTH = 1000
WARNING_TOKEN_LENGTH = 900
MIN_TURNS = 4
MAX_TURNS = 30

# Training.
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 2

# Checkpoints / logging.
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

SEED = 42

# False = fresh training.
# True = resume the latest checkpoint in OUTPUT_DIR.
RESUME_FROM_CHECKPOINT = False

REBUILD_CACHE = True

# True = report bad records and continue with valid records.
# False = stop if any malformed/invalid records are found.
ALLOW_VALID_RECORDS_WITH_BAD_LINES = True


# ==============================================================================
# TASK DEFINITIONS
# ==============================================================================

V5B_INSTRUCTION = "Predict assistant communication strategy"
V5A_INSTRUCTION = "Predict the assistant objective"
V4A_INSTRUCTION = "Extract long-term memories"
V4B_INSTRUCTION = "Extract long-term people memories"

TASK_NAMES = ("V5B", "V5A", "V4A", "V4B")

EXPECTED_V5B_FIELDS = {
    "tone",
    "communication_style",
    "detail_level",
    "approach",
}


# ==============================================================================
# SYSTEM PROMPTS
# ==============================================================================

V5B_SYSTEM_PROMPT = """
You are Compatifi V5B.

Your ONLY task is to predict the best communication strategy
for the assistant in the current conversation.

V5B predicts HOW the assistant should communicate.

V5B does NOT:
- generate the final assistant reply
- predict the assistant objective
- extract long-term memories
- extract people memories
- summarize the conversation
- predict user goals
- generate personality information
- predict future events

Return EXACTLY one JSON object containing EXACTLY these four keys:

{
  "tone": "...",
  "communication_style": "...",
  "detail_level": "...",
  "approach": "..."
}

Rules:
1. Select ONE tone.
2. Select ONE communication style.
3. Select ONE detail level.
4. Provide ONE communication approach.
5. Use only the provided relationship and conversation.
6. Do not generate the final assistant reply.
7. Do not predict V5A objectives.
8. Do not generate memories or people memories.
9. Do not summarize the conversation.
10. Do not add any fields.
11. Return JSON only.
""".strip()

V5A_SYSTEM_PROMPT = """
You are Compatifi V5A.

Your ONLY task is to predict the immediate objective that the assistant
should accomplish in the current conversation.

Do not generate an assistant reply.
Do not extract long-term memories.
Do not extract people memories.
Do not generate a conversation summary.

Return EXACTLY one JSON object:

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

Return JSON only.
""".strip()

V4A_SYSTEM_PROMPT = """
You are Compatifi V4A.

Perform the original V4A long-term memory extraction task.
Preserve the original V4A output schema and semantics.

Do not convert this task into V5A.
Do not convert this task into V5B.
Do not add V5A or V5B fields.

Return the original V4A target structure.
""".strip()

V4B_SYSTEM_PROMPT = """
You are Compatifi V4B.

Perform the original V4B long-term people-memory extraction task.
Preserve the original V4B output schema and semantics.

Do not convert this task into V5A.
Do not convert this task into V5B.
Do not add V5A or V5B fields.

Return the original V4B target structure.
""".strip()

SYSTEM_PROMPTS = {
    "V5B": V5B_SYSTEM_PROMPT,
    "V5A": V5A_SYSTEM_PROMPT,
    "V4A": V4A_SYSTEM_PROMPT,
    "V4B": V4B_SYSTEM_PROMPT,
}


# ==============================================================================
# HELPERS
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return normalize_json_value({
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    })


def instruction_to_task(instruction):
    instruction = instruction.strip()

    if instruction == V5B_INSTRUCTION:
        return "V5B"
    if instruction == V5A_INSTRUCTION:
        return "V5A"
    if instruction == V4A_INSTRUCTION:
        return "V4A"
    if instruction == V4B_INSTRUCTION:
        return "V4B"

    return "UNKNOWN"


def count_conversation_turns(conversation):
    if not isinstance(conversation, str):
        return 0

    return len([
        line
        for line in conversation.splitlines()
        if line.strip()
    ])


# ==============================================================================
# V5B VALIDATION
# ==============================================================================

def validate_v5b_record(item, index):
    input_data = item["input"]
    output = item["output"]

    if not isinstance(input_data, dict):
        raise ValueError("input must be an object")

    required_input = {
        "domain",
        "relationship",
        "conversation",
    }

    missing = required_input - set(input_data.keys())

    if missing:
        raise ValueError(
            f"missing input fields: {sorted(missing)}"
        )

    if input_data["domain"] != "relationship":
        raise ValueError(
            "domain must be 'relationship'"
        )

    if (
        not isinstance(input_data["relationship"], str)
        or not input_data["relationship"].strip()
    ):
        raise ValueError(
            "relationship is empty"
        )

    conversation = input_data["conversation"]

    if (
        not isinstance(conversation, str)
        or not conversation.strip()
    ):
        raise ValueError(
            "conversation is empty"
        )

    turns = count_conversation_turns(
        conversation
    )

    if turns < MIN_TURNS or turns > MAX_TURNS:
        raise ValueError(
            f"conversation has {turns} turns; "
            f"expected {MIN_TURNS}-{MAX_TURNS}"
        )

    if not isinstance(output, dict):
        raise ValueError(
            "output must be an object"
        )

    fields = set(output.keys())

    if fields != EXPECTED_V5B_FIELDS:
        extra = fields - EXPECTED_V5B_FIELDS
        missing = EXPECTED_V5B_FIELDS - fields

        raise ValueError(
            f"V5B schema mismatch; "
            f"extra={sorted(extra)}, "
            f"missing={sorted(missing)}"
        )

    for field in EXPECTED_V5B_FIELDS:
        if is_empty_value(output[field]):
            raise ValueError(
                f"{field} is empty"
            )

    return turns


# ==============================================================================
# JSONL LOADER
# ==============================================================================

def load_jsonl(
    path,
    expected_task,
    strict_v5b=False,
):
    path = Path(path)

    print()
    print("=" * 80)
    print(f"VALIDATING: {path}")
    print("=" * 80)

    if not path.exists():
        print(
            f"ERROR: required file does not exist: {path}"
        )
        sys.exit(1)

    records = []
    invalid = 0
    duplicates = 0
    fingerprints = set()
    turn_counts = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, raw_line in enumerate(
            file,
            1,
        ):
            if not raw_line.strip():
                continue

            try:
                item = json.loads(raw_line)

                if not isinstance(item, dict):
                    raise ValueError(
                        "record is not a JSON object"
                    )

                for field in (
                    "instruction",
                    "input",
                    "output",
                ):
                    if field not in item:
                        raise ValueError(
                            f"missing required field: {field}"
                        )

                if (
                    not isinstance(
                        item["instruction"],
                        str,
                    )
                    or not item["instruction"].strip()
                ):
                    raise ValueError(
                        "instruction is empty"
                    )

                if is_empty_value(item["input"]):
                    raise ValueError(
                        "input is empty"
                    )

                if is_empty_value(item["output"]):
                    raise ValueError(
                        "output is empty"
                    )

                task = instruction_to_task(
                    item["instruction"]
                )

                if task != expected_task:
                    raise ValueError(
                        f"expected {expected_task}, "
                        f"instruction maps to {task}"
                    )

                if strict_v5b:
                    turns = validate_v5b_record(
                        item,
                        len(records),
                    )
                    turn_counts.append(turns)

                fingerprint = record_fingerprint(
                    item
                )

                if fingerprint in fingerprints:
                    duplicates += 1
                    print(
                        f"[DUPLICATE] "
                        f"{path.name} "
                        f"line {line_number}"
                    )
                else:
                    fingerprints.add(
                        fingerprint
                    )

                item["_task"] = task
                records.append(item)

            except Exception as exc:
                invalid += 1
                print(
                    f"[INVALID] "
                    f"{path.name} "
                    f"line {line_number}: "
                    f"{exc}"
                )

    print(
        f"Valid records     : {len(records):,}"
    )
    print(
        f"Invalid records   : {invalid:,}"
    )
    print(
        f"Duplicate records : {duplicates:,}"
    )

    if strict_v5b and turn_counts:
        buckets = Counter()

        for turns in turn_counts:
            if 4 <= turns <= 8:
                buckets["4-8"] += 1
            elif 9 <= turns <= 15:
                buckets["9-15"] += 1
            elif 16 <= turns <= 24:
                buckets["16-24"] += 1
            elif 25 <= turns <= 30:
                buckets["25-30"] += 1

        print()
        print(
            "V5B CONVERSATION TURN DISTRIBUTION"
        )

        total_turn_records = len(turn_counts)

        for bucket in (
            "4-8",
            "9-15",
            "16-24",
            "25-30",
        ):
            count = buckets[bucket]
            percentage = (
                count
                / total_turn_records
                * 100
            )

            print(
                f"{bucket:>7}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

        print(
            f"Average turns: "
            f"{sum(turn_counts) / len(turn_counts):.2f}"
        )

    if invalid and not ALLOW_VALID_RECORDS_WITH_BAD_LINES:
        print(
            "ERROR: invalid records found; stopping."
        )
        sys.exit(1)

    if not records:
        print(
            "ERROR: no valid records found."
        )
        sys.exit(1)

    return (
        Dataset.from_list(records),
        {
            "valid": len(records),
            "invalid": invalid,
            "duplicates": duplicates,
        },
    )


# ==============================================================================
# STARTUP
# ==============================================================================

set_seed(SEED)

if not torch.cuda.is_available():
    print(
        "ERROR: CUDA is not available."
    )
    sys.exit(1)

if torch.cuda.is_bf16_supported():
    COMPUTE_DTYPE = torch.bfloat16
    USE_BF16 = True
    USE_FP16 = False
else:
    COMPUTE_DTYPE = torch.float16
    USE_BF16 = False
    USE_FP16 = True

print("=" * 80)
print("COMPATIFI V5B TRAINING")
print("=" * 80)
print(
    f"GPU: {torch.cuda.get_device_name(0)}"
)
print(
    f"Base model: {MODEL_NAME}"
)


# ==============================================================================
# LOAD DATASETS
# ==============================================================================

v5b_dataset, v5b_stats = load_jsonl(
    V5B_DATASET_PATH,
    "V5B",
    strict_v5b=True,
)

v5a_dataset, v5a_stats = load_jsonl(
    V5A_REPLAY_PATH,
    "V5A",
)

v4a_dataset, v4a_stats = load_jsonl(
    V4A_REPLAY_PATH,
    "V4A",
)

v4b_dataset, v4b_stats = load_jsonl(
    V4B_REPLAY_PATH,
    "V4B",
)


# ==============================================================================
# FULL DATASET MODE
# ==============================================================================

v5b_dataset = v5b_dataset.shuffle(
    seed=SEED
)
v5a_dataset = v5a_dataset.shuffle(
    seed=SEED + 1
)
v4a_dataset = v4a_dataset.shuffle(
    seed=SEED + 2
)
v4b_dataset = v4b_dataset.shuffle(
    seed=SEED + 3
)

combined_dataset = concatenate_datasets([
    v5b_dataset,
    v5a_dataset,
    v4a_dataset,
    v4b_dataset,
]).shuffle(
    seed=SEED
)

combined_counts = Counter(
    combined_dataset["_task"]
)

combined_total = len(
    combined_dataset
)

print()
print("=" * 80)
print("FULL DATASET DISTRIBUTION")
print("=" * 80)

for task in TASK_NAMES:
    count = combined_counts.get(
        task,
        0,
    )
    percentage = (
        count / combined_total * 100
    )

    print(
        f"{task}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )

print(
    f"TOTAL: {combined_total:,}"
)


# ==============================================================================
# TOKENIZER
# ==============================================================================

print()
print(
    "Loading tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ==============================================================================
# FORMAT DATASET
# ==============================================================================

def format_sample(example):
    task = instruction_to_task(
        example["instruction"]
    )

    if task not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown task: "
            f"{example['instruction']!r}"
        )

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

    user_content = (
        f"Task: "
        f"{example['instruction'].strip()}\n\n"
        f"Input:\n"
        f"{input_data}"
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

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ),
        "task": task,
    }


if (
    REBUILD_CACHE
    and os.path.exists(CACHE_DIR)
):
    shutil.rmtree(
        CACHE_DIR
    )

print()
print(
    "Formatting full dataset..."
)

formatted_dataset = combined_dataset.map(
    format_sample,
    remove_columns=(
        combined_dataset.column_names
    ),
    desc="Formatting V5B + replay",
)

formatted_dataset.save_to_disk(
    CACHE_DIR
)


# ==============================================================================
# TOKEN LENGTH CHECK
# ==============================================================================

print()
print("=" * 80)
print("TOKEN LENGTH SAMPLE CHECK")
print("=" * 80)

for task in TASK_NAMES:

    indices = [
        index
        for index, task_name in enumerate(
            formatted_dataset["task"]
        )
        if task_name == task
    ]

    if not indices:
        continue

    sample_count = min(
        1000,
        len(indices),
    )

    rng = random.Random(
        SEED + TASK_NAMES.index(task)
    )

    selected = rng.sample(
        indices,
        sample_count,
    )

    lengths = sorted([
        len(
            tokenizer.encode(
                formatted_dataset[index]["text"],
                add_special_tokens=True,
            )
        )
        for index in selected
    ])

    average = (
        sum(lengths) / len(lengths)
    )

    median = lengths[
        len(lengths) // 2
    ]

    p95 = lengths[
        min(
            len(lengths) - 1,
            int(len(lengths) * 0.95),
        )
    ]

    over_900 = sum(
        length > WARNING_TOKEN_LENGTH
        for length in lengths
    )

    over_1000 = sum(
        length > MAX_SEQ_LENGTH
        for length in lengths
    )

    print()
    print(task)
    print(
        f"  average      : {average:.2f}"
    )
    print(
        f"  median       : {median}"
    )
    print(
        f"  p95          : {p95}"
    )
    print(
        f"  maximum      : {max(lengths)}"
    )
    print(
        f"  >900 tokens  : {over_900}"
    )
    print(
        f"  >1000 tokens : {over_1000}"
    )


# ==============================================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ==============================================================================

def stratified_split(
    dataset,
    validation_ratio,
    seed,
):
    rng = random.Random(
        seed
    )

    train_parts = []
    eval_parts = []

    for task in TASK_NAMES:

        indices = [
            index
            for index, task_name in enumerate(
                dataset["task"]
            )
            if task_name == task
        ]

        rng.shuffle(indices)

        if len(indices) <= 1:
            evaluation_count = 0
        else:
            evaluation_count = max(
                1,
                min(
                    len(indices) - 1,
                    round(
                        len(indices)
                        * validation_ratio
                    ),
                ),
            )

        evaluation_indices = indices[
            :evaluation_count
        ]

        train_indices = indices[
            evaluation_count:
        ]

        train_parts.append(
            dataset.select(
                train_indices
            )
        )

        if evaluation_indices:
            eval_parts.append(
                dataset.select(
                    evaluation_indices
                )
            )

    train_dataset = concatenate_datasets(
        train_parts
    ).shuffle(
        seed=seed
    )

    eval_dataset = concatenate_datasets(
        eval_parts
    ).shuffle(
        seed=seed + 1
    )

    return (
        train_dataset,
        eval_dataset,
    )


train_dataset, eval_dataset = (
    stratified_split(
        formatted_dataset,
        VALIDATION_RATIO,
        SEED,
    )
)


def print_distribution(
    dataset,
    title,
):
    counts = Counter(
        dataset["task"]
    )

    total = len(dataset)

    print()
    print(title)
    print(
        f"Total: {total:,}"
    )

    for task in TASK_NAMES:
        count = counts.get(
            task,
            0,
        )

        percentage = (
            count / total * 100
        )

        print(
            f"  {task}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
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
# LOAD V5A MERGED MODEL
# ==============================================================================

print()
print("=" * 80)
print("LOADING V5A MERGED BASE MODEL")
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


# ==============================================================================
# TRL CONFIGURATION
# ==============================================================================

sft_parameters = inspect.signature(
    SFTConfig.__init__
).parameters

training_kwargs = {
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": (
        PER_DEVICE_BATCH_SIZE
    ),
    "per_device_eval_batch_size": (
        PER_DEVICE_BATCH_SIZE
    ),
    "gradient_accumulation_steps": (
        GRADIENT_ACCUMULATION_STEPS
    ),
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "bf16": USE_BF16,
    "fp16": USE_FP16,
    "tf32": True,
    "gradient_checkpointing": True,
    "logging_steps": LOGGING_STEPS,
    "logging_first_step": True,
    "report_to": "none",
    "save_strategy": "steps",
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "seed": SEED,
    "remove_unused_columns": True,
    "packing": False,
}

if "max_length" in sft_parameters:
    training_kwargs["max_length"] = (
        MAX_SEQ_LENGTH
    )
elif "max_seq_length" in sft_parameters:
    training_kwargs["max_seq_length"] = (
        MAX_SEQ_LENGTH
    )
else:
    raise RuntimeError(
        "TRL has neither max_length "
        "nor max_seq_length."
    )

if "eval_strategy" in sft_parameters:
    training_kwargs["eval_strategy"] = "steps"
    training_kwargs["eval_steps"] = SAVE_STEPS
elif "evaluation_strategy" in sft_parameters:
    training_kwargs["evaluation_strategy"] = "steps"
    training_kwargs["eval_steps"] = SAVE_STEPS

if "dataset_text_field" in sft_parameters:
    training_kwargs["dataset_text_field"] = "text"

training_args = SFTConfig(
    **training_kwargs
)


# ==============================================================================
# LORA
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
# SFT TRAINER
# ==============================================================================

trainer_signature = inspect.signature(
    SFTTrainer.__init__
).parameters

trainer_kwargs = {
    "model": model,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "args": training_args,
    "peft_config": lora_config,
}

if "processing_class" in trainer_signature:
    trainer_kwargs["processing_class"] = (
        tokenizer
    )
elif "tokenizer" in trainer_signature:
    trainer_kwargs["tokenizer"] = (
        tokenizer
    )
else:
    raise RuntimeError(
        "SFTTrainer has neither "
        "processing_class nor tokenizer."
    )

trainer = SFTTrainer(
    **trainer_kwargs
)


# ==============================================================================
# CHECKPOINT / TRAINING
# ==============================================================================

last_checkpoint = None

if os.path.isdir(
    OUTPUT_DIR
):
    last_checkpoint = (
        trainer_utils.get_last_checkpoint(
            OUTPUT_DIR
        )
    )

print()
print("=" * 80)
print("STARTING COMPATIFI V5B TRAINING")
print("=" * 80)

print(
    f"V5B samples : "
    f"{combined_counts.get('V5B', 0):,}"
)
print(
    f"V5A replay  : "
    f"{combined_counts.get('V5A', 0):,}"
)
print(
    f"V4A replay  : "
    f"{combined_counts.get('V4A', 0):,}"
)
print(
    f"V4B replay  : "
    f"{combined_counts.get('V4B', 0):,}"
)
print(
    f"Total       : "
    f"{combined_total:,}"
)
print(
    f"Epochs      : "
    f"{NUM_TRAIN_EPOCHS}"
)
print(
    f"Effective batch: "
    f"{PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)
print(
    f"Max sequence length: "
    f"{MAX_SEQ_LENGTH}"
)

print("=" * 80)

if (
    RESUME_FROM_CHECKPOINT
    and last_checkpoint
):
    train_result = trainer.train(
        resume_from_checkpoint=(
            last_checkpoint
        )
    )
else:
    train_result = trainer.train()


# ==============================================================================
# SAVE FINAL MODEL
# ==============================================================================

print()
print("=" * 80)
print("SAVING FINAL V5B MODEL")
print("=" * 80)

os.makedirs(
    FINAL_MODEL_DIR,
    exist_ok=True,
)

trainer.save_model(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)

print()
print("=" * 80)
print("V5B FULL-DATASET TRAINING COMPLETE")
print("=" * 80)

print(
    f"Final model: "
    f"{FINAL_MODEL_DIR}"
)

if hasattr(
    train_result,
    "metrics",
):
    print()
    print("FINAL METRICS")

    for key, value in (
        train_result.metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

print("=" * 80)
