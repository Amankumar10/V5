#!/usr/bin/env python3

# ============================================================
# COMPATIFI V5D TRAINING
# Conversation Understanding Model
#
# PIPELINE:
# V4B -> V5A -> V5B -> V5C -> V5D
#
# IMPORTANT CONTINUAL LEARNING DESIGN:
# V5D starts from the correctly MERGED V5C model.
# Uses the same proven SFTTrainer + QLoRA architecture as V5B/V5C.
#
# V5D DATASET ONLY - No replay dataset.
# ============================================================

import json
import os
import sys

import torch
from datasets import Dataset

from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    trainer_utils,
)


# ============================================================
# 1. GPU SETTINGS
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available.")
    sys.exit(1)

print("=" * 80)
print("COMPATIFI V5D TRAINING")
print("=" * 80)
print()

print("GPU:", torch.cuda.get_device_name(0))

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

print(f"GPU Memory: {gpu_memory:.1f} GB")
print()


# ============================================================
# 2. PATHS
# ============================================================

# MUST be the correctly merged V5C model.
# Do NOT use a V5C LoRA adapter directory.

BASE_CHECKPOINT = "./V5C_Final_Merged_Model"

# V5D dataset
V5D_DATASET = "./datasets/V5D/V5D.jsonl"

# Normalized dataset output
NORMALIZED_DATASET = "./v5d_normalized_training.jsonl"

# Training output
OUTPUT_DIR = "./V5D_Final"


# ============================================================
# 3. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are Compatifi V5D.

Your task is to understand the current state of a conversation
and extract structured information from the provided context.

Analyze only the information explicitly supported by the
conversation and input context.

Identify:
- the main topic
- the user's current situation
- the user's current goal
- the user's current issue or obstacle
- the current conversation stage
- the help required
- the type of help required
- any relevant risks
- confidence based on available information

Important rules:
- use only information supported by the provided context
- do not invent facts
- do not assume missing information
- do not create unsupported user profiles
- do not infer long-term memories
- do not provide advice unless explicitly requested
- do not answer the user's conversation directly
- do not generate a normal assistant reply
- focus only on structured conversation understanding

If information is unknown or unsupported, represent it according
to the output format demonstrated in the training examples.

Return only the requested structured output.
"""


# ============================================================
# 4. TRAINING PARAMETERS
# ============================================================

MAX_SEQ_LENGTH = 1024

# L40S optimized
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8

# Conservative specialization
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1

WARMUP_RATIO = 0.05

SAVE_STEPS = 250
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2


# ============================================================
# 5. LoRA CONFIGURATION
# Same architecture as V5B / V5C
# ============================================================

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


# ============================================================
# 6. PRECISION
# ============================================================

if torch.cuda.is_bf16_supported():
    COMPUTE_DTYPE = torch.bfloat16
    USE_BF16 = True
    USE_FP16 = False
else:
    COMPUTE_DTYPE = torch.float16
    USE_BF16 = False
    USE_FP16 = True

print("Compute dtype:", COMPUTE_DTYPE)
print()


# ============================================================
# 7. CHECK FILES
# ============================================================

print("=" * 80)
print("CHECKING FILES")
print("=" * 80)
print()

print("V5C base model:", os.path.abspath(BASE_CHECKPOINT))
print("V5D dataset:", os.path.abspath(V5D_DATASET))
print("Output directory:", os.path.abspath(OUTPUT_DIR))
print()

if not os.path.isdir(BASE_CHECKPOINT):
    raise FileNotFoundError(
        "V5C merged model was not found:\n"
        + os.path.abspath(BASE_CHECKPOINT)
    )

if not os.path.isfile(V5D_DATASET):
    raise FileNotFoundError(
        "V5D dataset was not found:\n"
        + os.path.abspath(V5D_DATASET)
    )


# ============================================================
# 8. VERIFY V5C IS A MERGED MODEL
# ============================================================

print("=" * 80)
print("CHECKING V5C MODEL")
print("=" * 80)
print()

v5c_adapter_config = os.path.join(
    BASE_CHECKPOINT,
    "adapter_config.json",
)

if os.path.exists(v5c_adapter_config):
    print("WARNING:")
    print("adapter_config.json exists in the V5C directory.")
    print("Make sure this is the MERGED V5C model.")
    print("Do not use the V5C LoRA adapter directory.")
else:
    print("V5C appears to be a normal/merged model directory.")

print()


# ============================================================
# 9. LOAD TOKENIZER
# ============================================================

print("=" * 80)
print("LOADING V5C TOKENIZER")
print("=" * 80)
print()

tokenizer = AutoTokenizer.from_pretrained(
    BASE_CHECKPOINT,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Tokenizer loaded.")
print()


# ============================================================
# 10. LOAD V5C MODEL
# ============================================================

print("=" * 80)
print("LOADING V5C MODEL")
print("=" * 80)
print()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_CHECKPOINT,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

print("V5C model loaded.")
print()


# ============================================================
# 11. LoRA CONFIGURATION
# ============================================================

print("=" * 80)
print("CONFIGURING V5D LoRA")
print("=" * 80)
print()

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)

print("LoRA rank:", LORA_R)
print("LoRA alpha:", LORA_ALPHA)
print()


# ============================================================
# 12. LOAD V5D DATASET
# ============================================================

print("=" * 80)
print("LOADING V5D DATASET")
print("=" * 80)
print()

raw_dataset = Dataset.from_json(V5D_DATASET)

print(f"V5D samples: {len(raw_dataset):,}")
print()

if len(raw_dataset) == 0:
    raise ValueError("V5D dataset is empty.")


# ============================================================
# 13. VALIDATE DATASET
# ============================================================

print("=" * 80)
print("VALIDATING V5D DATASET")
print("=" * 80)
print()

for index, example in enumerate(raw_dataset):

    if "input" not in example:
        raise ValueError(
            f"Example {index + 1} missing 'input'."
        )

    if "output" not in example:
        raise ValueError(
            f"Example {index + 1} missing 'output'."
        )

    if not isinstance(example["input"], dict):
        raise ValueError(
            f"Example {index + 1}: 'input' must be a JSON object."
        )

    if not isinstance(example["output"], dict):
        raise ValueError(
            f"Example {index + 1}: 'output' must be a JSON object."
        )

print("Dataset validation passed.")
print()


# ============================================================
# 14. JSON SERIALIZATION
# ============================================================

def dump_json(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ============================================================
# 15. FORMAT SAMPLE
# SAME CHAT PIPELINE AS V5B / V5C
# ============================================================

def format_sample(example):

    instruction = str(
        example.get(
            "instruction",
            "Analyze the conversation and extract structured understanding.",
        )
    ).strip()

    input_data = example.get("input", {})
    output_data = example.get("output", {})

    # Keep successful V5B/V5C style:
    # Input -> Instruction

    user_content = (
        "Input:\n"
        + dump_json(input_data)
        + "\n\n"
        + "Instruction:\n"
        + instruction
    )

    assistant_content = dump_json(output_data)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": assistant_content,
        },
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": formatted_text,
    }


# ============================================================
# 16. FORMAT DATASET
# ============================================================

print("=" * 80)
print("FORMATTING V5D DATASET")
print("=" * 80)
print()

formatted_dataset = raw_dataset.map(
    format_sample,
    remove_columns=raw_dataset.column_names,
    desc="Formatting V5D dataset",
)

print(
    "Formatted samples:",
    len(formatted_dataset),
)

print()

if len(formatted_dataset) == 0:
    raise ValueError(
        "No valid formatted samples remain."
    )


# ============================================================
# 17. SAVE NORMALIZED DATASET
# ============================================================

print("=" * 80)
print("SAVING NORMALIZED DATASET")
print("=" * 80)
print()

with open(
    NORMALIZED_DATASET,
    "w",
    encoding="utf-8",
) as f:

    for example in formatted_dataset:
        f.write(
            json.dumps(
                example,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    "Saved:",
    os.path.abspath(NORMALIZED_DATASET),
)

print()


# ============================================================
# 18. DATASET STATISTICS
# ============================================================

print("=" * 80)
print("V5D DATASET STATISTICS")
print("=" * 80)
print()

dataset_size = len(formatted_dataset)
sample_size = min(1000, dataset_size)

token_lengths = []
over_max_length = 0

for i in range(sample_size):

    text = formatted_dataset[i]["text"]

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    token_length = len(tokens)

    token_lengths.append(token_length)

    if token_length > MAX_SEQ_LENGTH:
        over_max_length += 1


if token_lengths:
    average_length = (
        sum(token_lengths)
        / len(token_lengths)
    )
    maximum_length = max(token_lengths)
    minimum_length = min(token_lengths)
else:
    average_length = 0
    maximum_length = 0
    minimum_length = 0


print(f"Dataset size         : {dataset_size:,}")
print(f"Statistics sample    : {sample_size:,}")
print(f"Minimum token length : {minimum_length}")
print(f"Average token length : {average_length:.2f}")
print(f"Maximum token length : {maximum_length}")
print(
    f"Over {MAX_SEQ_LENGTH} tokens "
    f"(sample) : {over_max_length:,}"
)

print()


# ============================================================
# 19. TRAINING CONFIGURATION
# ============================================================

print("=" * 80)
print("CREATING V5D TRAINING CONFIGURATION")
print("=" * 80)
print()

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    # Batch
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # Sequence
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,

    # Learning
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,

    # Optimizer
    optim="paged_adamw_8bit",

    # Precision
    bf16=USE_BF16,
    fp16=USE_FP16,
    tf32=True,

    # Gradient checkpointing
    gradient_checkpointing=True,

    # Logging
    logging_steps=LOGGING_STEPS,

    # Checkpoints
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,

    # Misc
    report_to="none",
    disable_tqdm=False,
)


# ============================================================
# 20. CREATE SFT TRAINER
# ============================================================

print("=" * 80)
print("CREATING V5D SFT TRAINER")
print("=" * 80)
print()

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("V5D SFTTrainer created.")
print()


# ============================================================
# 21. TRAINABLE PARAMETERS
# ============================================================

print("=" * 80)
print("TRAINABLE PARAMETERS")
print("=" * 80)
print()

trainer.model.print_trainable_parameters()

print()


# ============================================================
# 22. CHECK FOR EXISTING CHECKPOINT
# ============================================================

print("=" * 80)
print("CHECKING FOR EXISTING V5D CHECKPOINT")
print("=" * 80)
print()

last_checkpoint = None

if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = trainer_utils.get_last_checkpoint(
        OUTPUT_DIR
    )


if last_checkpoint:

    print("Checkpoint found:")
    print(os.path.abspath(last_checkpoint))
    print()
    print("Training will RESUME.")

else:

    print("No checkpoint found.")
    print("Starting fresh V5D training from V5C.")

print()


# ============================================================
# 23. FINAL CONFIGURATION DISPLAY
# ============================================================

print("=" * 80)
print("V5D TRAINING CONFIGURATION")
print("=" * 80)
print()

print("Base model       :", BASE_CHECKPOINT)
print("Training dataset :", V5D_DATASET)
print("Training samples :", f"{len(formatted_dataset):,}")
print("Epochs           :", NUM_TRAIN_EPOCHS)
print("Batch size       :", PER_DEVICE_BATCH_SIZE)
print("Gradient accum.  :", GRADIENT_ACCUMULATION_STEPS)
print("Learning rate    :", LEARNING_RATE)
print("Max sequence     :", MAX_SEQ_LENGTH)
print("Save steps       :", SAVE_STEPS)
print()
print("Trainer          : SFTTrainer")
print("Training style   : Full formatted chat sequence")
print("Manual masking   : NONE")
print("Replay dataset   : NONE")
print()
print("Architecture consistency:")
print("V5B -> SFTTrainer")
print("V5C -> SFTTrainer")
print("V5D -> SFTTrainer")
print()


# ============================================================
# 24. START TRAINING
# ============================================================

print("=" * 80)
print("STARTING V5D TRAINING")
print("=" * 80)
print()

if last_checkpoint:
    trainer.train(
        resume_from_checkpoint=last_checkpoint
    )
else:
    trainer.train()


# ============================================================
# 25. SAVE FINAL V5D MODEL
# ============================================================

FINAL_MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "final_model",
)

print()
print("=" * 80)
print("SAVING FINAL V5D MODEL")
print("=" * 80)
print()

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print()
print("Final V5D model:")
print(os.path.abspath(FINAL_MODEL_DIR))
print()


# ============================================================
# 26. COMPLETE
# ============================================================

print("=" * 80)
print("V5D TRAINING COMPLETE")
print("=" * 80)
print()

print("Pipeline:")
print("V4B -> V5A -> V5B -> V5C -> V5D")
print()

print("V5D task:")
print("Conversation -> Structured Understanding")
print()

print("Training architecture:")
print("QLoRA + SFTTrainer")
print()

print("Replay dataset: NONE")
print()

print("Final model:")
print(os.path.abspath(FINAL_MODEL_DIR))
print()

print("=" * 80)
print("DONE")
print("=" * 80)
