#!/usr/bin/env python3

# ============================================================
# Compatifi V5A Training
# Assistant Objective Prediction
#
# V5A DATASET ONLY
# No replay dataset
# No strict label validation
# No objective-label rejection
#
# ============================================================


# ============================================================
# SECTION 1: IMPORTS
# ============================================================

import json
import os
import sys

import torch

from datasets import Dataset

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)

from trl import (
    SFTConfig,
    SFTTrainer,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    trainer_utils,
)


# ============================================================
# SECTION 2: GPU SETTINGS
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if not torch.cuda.is_available():

    print("ERROR: CUDA is not available.")
    print("An NVIDIA GPU is required.")

    sys.exit(1)


print("=" * 80)
print("COMPATIFI V5A TRAINING")
print("=" * 80)

print()

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

print(
    f"GPU Memory: {gpu_memory:.1f} GB"
)

print()


# ============================================================
# SECTION 3: CONFIGURATION
# ============================================================

BASE_CHECKPOINT = "../V4B_Final_Merged_Model"

# ONLY training dataset
V5A_DATASET = "./v5a_main.jsonl"

# Normalized temporary V5A dataset
NORMALIZED_DATASET = "./v5a_normalized_training.jsonl"

# Final checkpoints
OUTPUT_DIR = "./v5a_full_checkpoints"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are Compatifi V5A.

Your task is to predict the assistant objective for the current
conversation.

Use the information provided in:

- Domain
- Relationship
- Conversation

Determine what the assistant should accomplish next.

Return ONLY the JSON object from the training example.

Do not add explanations outside the JSON object.
"""


# ============================================================
# SECTION 4: TRAINING PARAMETERS
# ============================================================

MAX_SEQ_LENGTH = 1024

PER_DEVICE_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4

NUM_TRAIN_EPOCHS = 2

SAVE_STEPS = 500

LOGGING_STEPS = 10


# ============================================================
# SECTION 5: LoRA CONFIGURATION
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
# SECTION 6: PRECISION
# ============================================================

if torch.cuda.is_bf16_supported():

    COMPUTE_DTYPE = torch.bfloat16

    USE_BF16 = True

    USE_FP16 = False

else:

    COMPUTE_DTYPE = torch.float16

    USE_BF16 = False

    USE_FP16 = True


print(
    "Compute dtype:",
    COMPUTE_DTYPE
)

print()


# ============================================================
# SECTION 7: CHECK FILES
# ============================================================

print("=" * 80)
print("CHECKING FILES")
print("=" * 80)

print()

print(
    "Base checkpoint :",
    BASE_CHECKPOINT
)

print(
    "V5A dataset     :",
    V5A_DATASET
)

print(
    "Normalized data :",
    NORMALIZED_DATASET
)

print(
    "Output directory:",
    OUTPUT_DIR
)

print()


if not os.path.isdir(BASE_CHECKPOINT):

    print("ERROR: Base checkpoint not found.")

    print()

    print(
        "Expected:",
        BASE_CHECKPOINT
    )

    print()

    print(
        "Current working directory:",
        os.getcwd()
    )

    sys.exit(1)


if not os.path.isfile(V5A_DATASET):

    print("ERROR: V5A dataset not found.")

    print()

    print(
        "Expected:",
        V5A_DATASET
    )

    sys.exit(1)


print("Base checkpoint: OK")
print("V5A dataset    : OK")

print()


# ============================================================
# SECTION 8: TOKENIZER
# ============================================================

print("=" * 80)
print("LOADING TOKENIZER")
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
# SECTION 9: MODEL
# ============================================================

print("=" * 80)
print("LOADING MODEL")
print("=" * 80)

print()

print("Loading model with 4-bit QLoRA...")


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


model = prepare_model_for_kbit_training(
    model
)


print("Model loaded.")

print()


# ============================================================
# SECTION 10: LoRA
# ============================================================

peft_config = LoraConfig(

    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=LORA_TARGET_MODULES,

)


# ============================================================
# SECTION 11: FORMAT ONE SAMPLE
# ============================================================

def format_sample(example):

    inp = example.get(
        "input",
        {}
    )


    domain = inp.get(
        "domain",
        ""
    )


    relationship = inp.get(
        "relationship",
        ""
    )


    conversation = inp.get(
        "conversation",
        ""
    )


    instruction = example.get(
        "instruction",
        "Predict the assistant objective"
    )


    output = example.get(
        "output",
        {}
    )


    user_content = (

        f"Domain: {domain}\n\n"

        f"Relationship: {relationship}\n\n"

        f"Conversation:\n"
        f"{conversation}\n\n"

        f"Instruction:\n"
        f"{instruction}"

    )


    assistant_content = json.dumps(

        output,

        ensure_ascii=False,

        separators=(",", ":"),

    )


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
        "text": formatted_text
    }


# ============================================================
# SECTION 12: LOAD V5A DATASET
# ============================================================

print("=" * 80)
print("LOADING V5A DATASET")
print("=" * 80)

print()


raw_dataset = Dataset.from_json(
    V5A_DATASET
)


print(
    f"V5A samples: {len(raw_dataset):,}"
)

print()


# ============================================================
# SECTION 13: CREATE NORMALIZED TEMPORARY DATASET
# ============================================================

print("=" * 80)
print("CREATING NORMALIZED V5A DATASET")
print("=" * 80)

print()


formatted_dataset = raw_dataset.map(

    format_sample,

    remove_columns=raw_dataset.column_names,

    desc="Formatting V5A dataset",

)


print(
    "Formatted samples:",
    len(formatted_dataset)
)

print()


# ------------------------------------------------------------
# Save temporary normalized JSONL
# ------------------------------------------------------------

print(
    "Saving:",
    NORMALIZED_DATASET
)


with open(
    NORMALIZED_DATASET,
    "w",
    encoding="utf-8",
) as f:

    for item in formatted_dataset:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "Normalized dataset saved."
)

print()


# ============================================================
# SECTION 14: DATASET STATISTICS
# ============================================================

print("=" * 80)
print("DATASET STATISTICS")
print("=" * 80)

print()


dataset_size = len(
    formatted_dataset
)


sample_size = min(
    1000,
    dataset_size
)


lengths = []


for i in range(sample_size):

    text = formatted_dataset[i]["text"]

    length = len(
        tokenizer.encode(text)
    )

    lengths.append(length)


if lengths:

    average_length = (
        sum(lengths) / len(lengths)
    )

    maximum_length = max(lengths)

else:

    average_length = 0

    maximum_length = 0


print(
    f"Dataset size         : {dataset_size:,}"
)

print(
    f"Statistics sample    : {sample_size:,}"
)

print(
    f"Average token length : {average_length:.2f}"
)

print(
    f"Maximum token length : {maximum_length}"
)

print()


# ============================================================
# SECTION 15: TRAINING CONFIG
# ============================================================

print("=" * 80)
print("CREATING TRAINING CONFIGURATION")
print("=" * 80)

print()


training_args = SFTConfig(

    output_dir=OUTPUT_DIR,


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    per_device_train_batch_size=
        PER_DEVICE_BATCH_SIZE,


    gradient_accumulation_steps=
        GRADIENT_ACCUMULATION_STEPS,


    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    max_seq_length=MAX_SEQ_LENGTH,

    dataset_text_field="text",

    packing=False,


    # --------------------------------------------------------
    # Learning
    # --------------------------------------------------------

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_TRAIN_EPOCHS,

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="paged_adamw_8bit",


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16=USE_BF16,

    fp16=USE_FP16,

    tf32=True,


    # --------------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------------

    gradient_checkpointing=True,


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=LOGGING_STEPS,


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=SAVE_STEPS,

    save_total_limit=2,


    # --------------------------------------------------------
    # Misc
    # --------------------------------------------------------

    report_to="none",

    disable_tqdm=False,

)


# ============================================================
# SECTION 16: TRAINER
# ============================================================

trainer = SFTTrainer(

    model=model,

    train_dataset=formatted_dataset,

    args=training_args,

    peft_config=peft_config,

    processing_class=tokenizer,

)


# ============================================================
# SECTION 17: RESUME CHECK
# ============================================================

print("=" * 80)
print("CHECKING FOR EXISTING CHECKPOINT")
print("=" * 80)

print()


last_checkpoint = None


if os.path.isdir(OUTPUT_DIR):

    last_checkpoint = trainer_utils.get_last_checkpoint(
        OUTPUT_DIR
    )


if last_checkpoint:

    print(
        "Checkpoint found:"
    )

    print(
        last_checkpoint
    )

    print()

    print(
        "Training will RESUME."
    )

else:

    print(
        "No checkpoint found."
    )

    print(
        "Starting fresh V5A training."
    )


print()


# ============================================================
# SECTION 18: TRAIN
# ============================================================

print("=" * 80)
print("STARTING V5A TRAINING")
print("=" * 80)

print()

print(
    "Base checkpoint :",
    BASE_CHECKPOINT
)

print(
    "Training data   :",
    V5A_DATASET
)

print(
    "Replay data     : NONE"
)

print(
    "Training samples:",
    f"{len(formatted_dataset):,}"
)

print(
    "Epochs          :",
    NUM_TRAIN_EPOCHS
)

print(
    "Batch size      :",
    PER_DEVICE_BATCH_SIZE
)

print(
    "Gradient accum. :",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Learning rate   :",
    LEARNING_RATE
)

print()


trainer.train(
    resume_from_checkpoint=last_checkpoint
)


# ============================================================
# SECTION 19: SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_DIR = os.path.join(

    OUTPUT_DIR,

    "final_model"

)


print()
print("=" * 80)
print("SAVING FINAL V5A MODEL")
print("=" * 80)

print()


trainer.save_model(
    FINAL_MODEL_DIR
)


tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)


print()

print(
    "Final model:"
)

print(
    FINAL_MODEL_DIR
)


# ============================================================
# SECTION 20: COMPLETE
# ============================================================

print()
print("=" * 80)
print("V5A TRAINING COMPLETE")
print("=" * 80)

print()

print(
    "Training dataset:",
    V5A_DATASET
)

print(
    "Temporary dataset:",
    NORMALIZED_DATASET
)

print(
    "Final checkpoint:",
    FINAL_MODEL_DIR
)

print()

print("Replay dataset: NONE")

print()

print("=" * 80)
print("DONE")
print("=" * 80)