#!/usr/bin/env python3

# ============================================================
# COMPATIFI V5C TRAINING
# Conversation Summary Model
#
# PIPELINE:
#
# V4B -> V5A -> V5B -> V5C
#
# V5C DATASET ONLY
# No replay dataset
# No V4/V5A/V5B dataset replay
#
# HARDWARE:
# RTX 3060 12GB
#
# TRAINING:
# QLoRA 4-bit
# TRL SFTTrainer
# Same training architecture as V5A/V5B
#
# IMPORTANT:
# V5C starts from the MERGED V5B model.
# ============================================================


# ============================================================
# 1. IMPORTS
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
# 2. GPU SETTINGS
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if not torch.cuda.is_available():

    print("ERROR: CUDA is not available.")
    print("An NVIDIA GPU is required.")

    sys.exit(1)


print("=" * 80)
print("COMPATIFI V5C TRAINING")
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
# 3. PATHS
# ============================================================

# IMPORTANT:
# This MUST be the correctly merged V5B model.

BASE_CHECKPOINT = (
    r"E:\Compatifi-Model\V5B_Final_Merged_Model"
)


# ONLY V5C DATASET

V5C_DATASET = (
    r".\datasets\V5C\V5C.jsonl"
)


# Temporary normalized dataset

NORMALIZED_DATASET = (
    r".\v5c_normalized_training.jsonl"
)


# V5C checkpoints

OUTPUT_DIR = (
    r".\V5C_Final"
)


# ============================================================
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are Compatifi V5C.

Your task is to summarize the current relationship conversation.

Generate ONE concise, factual summary.

Use only information supported by the conversation.

Focus on:
- the main topic
- important facts
- the user's current situation
- the user's goal when relevant
- important problems or obstacles
- important outcomes when relevant

Do not:
- generate a reply
- give advice
- provide coaching
- calculate compatibility
- create a user profile
- create a people profile
- extract long-term memories
- invent information
- add unsupported assumptions
- add opinions
- include unnecessary details

If information is unknown, do not invent it.

Return only the summary.
"""


# ============================================================
# 5. TRAINING PARAMETERS
# ============================================================

MAX_SEQ_LENGTH = 1024


# RTX 3060 12GB

PER_DEVICE_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 16


# IMPORTANT:
# Lower than V5A/V5B to reduce aggressive specialization.

LEARNING_RATE = 5e-5


# Start with 1 epoch.
#
# With 75k samples this is already a large amount
# of training for a specialized V5C task.

NUM_TRAIN_EPOCHS = 2


WARMUP_RATIO = 0.05


SAVE_STEPS = 150

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# ============================================================
# 6. LoRA CONFIGURATION
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
# 7. PRECISION
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
# 8. CHECK FILES
# ============================================================

print("=" * 80)
print("CHECKING FILES")
print("=" * 80)

print()

print(
    "V5B base model:",
    os.path.abspath(BASE_CHECKPOINT)
)

print(
    "V5C dataset:",
    os.path.abspath(V5C_DATASET)
)

print(
    "Output directory:",
    os.path.abspath(OUTPUT_DIR)
)

print()


if not os.path.isdir(BASE_CHECKPOINT):

    raise FileNotFoundError(
        "\nV5B merged model was not found:\n"
        + os.path.abspath(BASE_CHECKPOINT)
    )


if not os.path.isfile(V5C_DATASET):

    raise FileNotFoundError(
        "\nV5C dataset was not found:\n"
        + os.path.abspath(V5C_DATASET)
    )


# ============================================================
# 9. VERIFY THAT V5B IS A MERGED MODEL
# ============================================================

print("=" * 80)
print("CHECKING V5B MODEL")
print("=" * 80)

print()


v5b_adapter_config = os.path.join(
    BASE_CHECKPOINT,
    "adapter_config.json"
)


if os.path.exists(v5b_adapter_config):

    print(
        "WARNING:"
    )

    print(
        "adapter_config.json exists in the V5B directory."
    )

    print(
        "Make sure this directory is your MERGED V5B model."
    )

    print(
        "Do not accidentally use the V5B LoRA adapter directory."
    )

else:

    print(
        "V5B appears to be a normal/merged model directory."
    )


print()


# ============================================================
# 10. LOAD TOKENIZER
# ============================================================

print("=" * 80)
print("LOADING V5B TOKENIZER")
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
# 11. LOAD V5B MODEL
# ============================================================

print("=" * 80)
print("LOADING V5B MODEL")
print("=" * 80)

print()

print(
    "Loading V5B with 4-bit QLoRA..."
)


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


print()

print("V5B model loaded.")

print()


# ============================================================
# 12. LoRA
# ============================================================

print("=" * 80)
print("CONFIGURING V5C LoRA")
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


print(
    "LoRA rank:",
    LORA_R
)

print(
    "LoRA alpha:",
    LORA_ALPHA
)

print()


# ============================================================
# 13. LOAD V5C DATASET
# ============================================================

print("=" * 80)
print("LOADING V5C DATASET")
print("=" * 80)

print()


raw_dataset = Dataset.from_json(
    V5C_DATASET
)


print(
    f"V5C samples: {len(raw_dataset):,}"
)

print()


if len(raw_dataset) == 0:

    raise ValueError(
        "V5C dataset is empty."
    )


# ============================================================
# 14. VALIDATE DATASET
# ============================================================

print("=" * 80)
print("VALIDATING V5C DATASET")
print("=" * 80)

print()


for index, example in enumerate(
    raw_dataset.select(
        range(min(100, len(raw_dataset))
    ))
):

    if "input" not in example:

        raise ValueError(
            f"Example {index + 1} missing 'input'."
        )


    if "output" not in example:

        raise ValueError(
            f"Example {index + 1} missing 'output'."
        )


    input_data = example["input"]

    output_data = example["output"]


    if "relationship" not in input_data:

        raise ValueError(
            f"Example {index + 1} missing "
            "'input.relationship'."
        )


    if "conversation" not in input_data:

        raise ValueError(
            f"Example {index + 1} missing "
            "'input.conversation'."
        )


    if "summary" not in output_data:

        raise ValueError(
            f"Example {index + 1} missing "
            "'output.summary'."
        )


print(
    "Dataset validation passed."
)

print()


# ============================================================
# 15. FORMAT SAMPLE
# ============================================================

def format_sample(example):

    input_data = example.get(
        "input",
        {}
    )


    relationship = str(
        input_data.get(
            "relationship",
            ""
        )
    ).strip()


    conversation = str(
        input_data.get(
            "conversation",
            ""
        )
    ).strip()


    instruction = str(
        example.get(
            "instruction",
            "Summarize the conversation."
        )
    ).strip()


    output_data = example.get(
        "output",
        {}
    )


    summary = str(
        output_data.get(
            "summary",
            ""
        )
    ).strip()


    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    user_content = (

        "Relationship:\n"
        + relationship
        + "\n\n"

        "Conversation:\n"
        + conversation
        + "\n\n"

        "Instruction:\n"
        + instruction

    )


    # --------------------------------------------------------
    # ASSISTANT OUTPUT
    # --------------------------------------------------------

    assistant_content = summary


    # --------------------------------------------------------
    # CHAT FORMAT
    # --------------------------------------------------------

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
# 16. FORMAT DATASET
# ============================================================

print("=" * 80)
print("FORMATTING V5C DATASET")
print("=" * 80)

print()


formatted_dataset = raw_dataset.map(

    format_sample,

    remove_columns=raw_dataset.column_names,

    desc="Formatting V5C dataset",

)


print(
    "Formatted samples:",
    len(formatted_dataset)
)

print()


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
    os.path.abspath(NORMALIZED_DATASET)
)

print()


# ============================================================
# 18. DATASET STATISTICS
# ============================================================

print("=" * 80)
print("V5C DATASET STATISTICS")
print("=" * 80)

print()


dataset_size = len(
    formatted_dataset
)


sample_size = min(
    1000,
    dataset_size
)


token_lengths = []


for i in range(sample_size):

    text = formatted_dataset[i]["text"]


    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )


    token_lengths.append(
        len(tokens)
    )


if token_lengths:

    average_length = (

        sum(token_lengths)
        /
        len(token_lengths)

    )

    maximum_length = max(
        token_lengths
    )

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
# 19. TRAINING CONFIGURATION
# ============================================================

print("=" * 80)
print("CREATING V5C TRAINING CONFIGURATION")
print("=" * 80)

print()


training_args = SFTConfig(

    output_dir=OUTPUT_DIR,


    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    per_device_train_batch_size=
        PER_DEVICE_BATCH_SIZE,

    gradient_accumulation_steps=
        GRADIENT_ACCUMULATION_STEPS,


    # --------------------------------------------------------
    # SEQUENCE
    # --------------------------------------------------------

    max_seq_length=MAX_SEQ_LENGTH,

    dataset_text_field="text",

    packing=False,


    # --------------------------------------------------------
    # LEARNING
    # --------------------------------------------------------

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_TRAIN_EPOCHS,

    lr_scheduler_type="cosine",

    warmup_ratio=WARMUP_RATIO,


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optim="paged_adamw_8bit",


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    bf16=USE_BF16,

    fp16=USE_FP16,

    tf32=True,


    # --------------------------------------------------------
    # GRADIENT CHECKPOINTING
    # --------------------------------------------------------

    gradient_checkpointing=True,


    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logging_steps=LOGGING_STEPS,


    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=SAVE_STEPS,

    save_total_limit=SAVE_TOTAL_LIMIT,


    # --------------------------------------------------------
    # MISC
    # --------------------------------------------------------

    report_to="none",

    disable_tqdm=False,

)


# ============================================================
# 20. CREATE TRAINER
# ============================================================

print("=" * 80)
print("CREATING V5C SFT TRAINER")
print("=" * 80)

print()


trainer = SFTTrainer(

    model=model,

    train_dataset=formatted_dataset,

    args=training_args,

    peft_config=peft_config,

    processing_class=tokenizer,

)


print(
    "V5C SFTTrainer created."
)

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
# 22. CHECK FOR CHECKPOINT
# ============================================================

print("=" * 80)
print("CHECKING FOR EXISTING V5C CHECKPOINT")
print("=" * 80)

print()


last_checkpoint = None


if os.path.isdir(OUTPUT_DIR):

    last_checkpoint = (
        trainer_utils.get_last_checkpoint(
            OUTPUT_DIR
        )
    )


if last_checkpoint:

    print(
        "Checkpoint found:"
    )

    print(
        os.path.abspath(
            last_checkpoint
        )
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
        "Starting fresh V5C training from V5B."
    )


print()


# ============================================================
# 23. FINAL TRAINING CONFIG DISPLAY
# ============================================================

print("=" * 80)
print("V5C TRAINING CONFIGURATION")
print("=" * 80)

print()

print(
    "Base model       :",
    BASE_CHECKPOINT
)

print(
    "Training dataset :",
    V5C_DATASET
)

print(
    "Training samples  :",
    f"{len(formatted_dataset):,}"
)

print(
    "Epochs            :",
    NUM_TRAIN_EPOCHS
)

print(
    "Batch size        :",
    PER_DEVICE_BATCH_SIZE
)

print(
    "Gradient accum.   :",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Learning rate     :",
    LEARNING_RATE
)

print(
    "Max sequence      :",
    MAX_SEQ_LENGTH
)

print(
    "Save steps        :",
    SAVE_STEPS
)

print()

print(
    "Replay dataset    : NONE"
)

print(
    "V4/V5A/V5B data   : NONE"
)

print()


# ============================================================
# 24. START TRAINING
# ============================================================

print("=" * 80)
print("STARTING V5C TRAINING")
print("=" * 80)

print()


if last_checkpoint:

    trainer.train(

        resume_from_checkpoint=
            last_checkpoint

    )

else:

    trainer.train()


# ============================================================
# 25. SAVE FINAL V5C
# ============================================================

FINAL_MODEL_DIR = os.path.join(

    OUTPUT_DIR,

    "final_model"

)


print()

print("=" * 80)
print("SAVING FINAL V5C MODEL")
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
    "Final V5C model:"
)

print(
    os.path.abspath(
        FINAL_MODEL_DIR
    )
)

print()


# ============================================================
# 26. COMPLETE
# ============================================================

print("=" * 80)
print("V5C TRAINING COMPLETE")
print("=" * 80)

print()

print(
    "Pipeline:"
)

print(
    "V4B -> V5A -> V5B -> V5C"
)

print()

print(
    "V5C task:"
)

print(
    "Conversation -> concise factual summary"
)

print()

print(
    "Training dataset:"
)

print(
    V5C_DATASET
)

print()

print(
    "Replay dataset: NONE"
)

print()

print(
    "Final model:"
)

print(
    os.path.abspath(
        FINAL_MODEL_DIR
    )
)

print()

print("=" * 80)
print("DONE")
print("=" * 80)