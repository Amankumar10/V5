# ============================================================
# COMPATIFI V5C - CONVERSATION SUMMARY MODEL
# ============================================================
#
# PURPOSE:
#   Train V5C to generate ONE concise factual summary
#   from a relationship conversation.
#
# TRAINING:
#   V5B model -> V5C dataset -> V5C model
#
# DATASET:
#   ONLY V5C dataset
#
# NO:
#   - V4A dataset
#   - V4B dataset
#   - V5B dataset
#   - replay dataset
#   - reply dataset
#
# HARDWARE:
#   NVIDIA RTX 3060 12GB
#
# FEATURES:
#   - QLoRA 4-bit
#   - RTX 3060 friendly
#   - Automatically finds latest checkpoint
#   - Resumes latest checkpoint if available
#   - Starts fresh from V5B if no checkpoint exists
#   - Supports instruction/input/output V5C JSONL
#   - Assistant-only loss
#   - Maximum sequence length 1024
#   - Saves tokenizer
#   - Saves final V5C model
# ============================================================


import os
import json
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    PeftModel,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 1. PATHS
# ============================================================

# IMPORTANT:
# Use a raw Windows string so \ does not cause path problems.

V5B_MODEL = r"E:\Compatifi-Model\V5B_Final_Merged_Model"

V5C_DATASET = r".\datasets\V5C\V5C.jsonl"

OUTPUT_DIR = r".\V5C_Final"


# ============================================================
# 2. TRAINING SETTINGS
# ============================================================

# RTX 3060 12GB
MAX_LENGTH = 1024

PER_DEVICE_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 16

NUM_EPOCHS = 2

LEARNING_RATE = 2e-4

WARMUP_RATIO = 0.05

SAVE_STEPS = 250

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# ============================================================
# 3. PRINT HEADER
# ============================================================

print("=" * 100)
print("COMPATIFI V5C TRAINING")
print("=" * 100)

print()
print("Task:")
print("Conversation -> Concise factual summary")

print()
print("Training flow:")
print("V5B -> V5C")

print()
print("Dataset:")
print("V5C ONLY")

print()
print("Replay dataset:")
print("NONE")

print()
print("Previous-stage dataset:")
print("NONE")


# ============================================================
# 4. GPU CHECK
# ============================================================

print("\n" + "=" * 100)
print("GPU CHECK")
print("=" * 100)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not detected.\n"
        "This training script requires an NVIDIA CUDA GPU."
    )

gpu_name = torch.cuda.get_device_name(0)

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

print("GPU:", gpu_name)
print("VRAM:", round(gpu_memory, 2), "GB")


# ============================================================
# 5. PATH CHECK
# ============================================================

print("\n" + "=" * 100)
print("PATH CHECK")
print("=" * 100)

print("V5B model:")
print(os.path.abspath(V5B_MODEL))

print()
print("V5C dataset:")
print(os.path.abspath(V5C_DATASET))

print()
print("V5C output:")
print(os.path.abspath(OUTPUT_DIR))


if not os.path.exists(V5B_MODEL):
    raise FileNotFoundError(
        "\nV5B model was not found:\n"
        + os.path.abspath(V5B_MODEL)
    )


if not os.path.isfile(V5C_DATASET):
    raise FileNotFoundError(
        "\nV5C dataset was not found:\n"
        + os.path.abspath(V5C_DATASET)
    )


# ============================================================
# 6. CHECK DATASET FILE
# ============================================================

print("\n" + "=" * 100)
print("CHECKING V5C DATASET")
print("=" * 100)

with open(
    V5C_DATASET,
    "r",
    encoding="utf-8"
) as f:

    lines = [
        line
        for line in f
        if line.strip()
    ]


print("Dataset examples:", len(lines))

if len(lines) == 0:
    raise ValueError("V5C dataset is empty.")


# ============================================================
# 7. VALIDATE FIRST DATASET EXAMPLES
# ============================================================

print("\nValidating V5C format...")

for index, line in enumerate(lines[:10]):

    try:
        item = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON at dataset line {index + 1}: {e}"
        )

    if "input" not in item:
        raise ValueError(
            f"V5C example {index + 1} is missing 'input'."
        )

    if "output" not in item:
        raise ValueError(
            f"V5C example {index + 1} is missing 'output'."
        )

    input_data = item["input"]
    output_data = item["output"]

    if "relationship" not in input_data:
        raise ValueError(
            f"V5C example {index + 1} is missing "
            "'input.relationship'."
        )

    if "conversation" not in input_data:
        raise ValueError(
            f"V5C example {index + 1} is missing "
            "'input.conversation'."
        )

    if "summary" not in output_data:
        raise ValueError(
            f"V5C example {index + 1} is missing "
            "'output.summary'."
        )


print("✅ V5C dataset format looks correct.")


# ============================================================
# 8. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 100)
print("LOADING V5B TOKENIZER")
print("=" * 100)

tokenizer = AutoTokenizer.from_pretrained(
    V5B_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Tokenizer loaded.")


# ============================================================
# 9. 4-BIT QLoRA CONFIG
# ============================================================

print("\n" + "=" * 100)
print("CONFIGURING 4-BIT QUANTIZATION")
print("=" * 100)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 10. LOAD V5B MODEL
# ============================================================

print("\n" + "=" * 100)
print("LOADING V5B MODEL")
print("=" * 100)

model = AutoModelForCausalLM.from_pretrained(
    V5B_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

    trust_remote_code=True,
)

print("✅ V5B model loaded.")


# ============================================================
# 11. MODEL CONFIG
# ============================================================

model.config.use_cache = False


# ============================================================
# 12. PREPARE FOR QLoRA
# ============================================================

print("\n" + "=" * 100)
print("PREPARING MODEL FOR QLoRA")
print("=" * 100)

model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_enable()


# ============================================================
# 13. V5C LoRA CONFIGURATION
# ============================================================

print("\n" + "=" * 100)
print("CONFIGURING V5C LoRA")
print("=" * 100)

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


# ============================================================
# 14. HANDLE V5B TYPE
# ============================================================
#
# If V5B is already a PEFT adapter:
#     load it and continue training.
#
# If V5B is a merged/full model:
#     attach a new V5C LoRA adapter.
#
# ============================================================

v5b_adapter_config = os.path.join(
    V5B_MODEL,
    "adapter_config.json"
)


if os.path.exists(v5b_adapter_config):

    print()
    print("Detected PEFT adapter in V5B.")

    print("Loading V5B adapter as trainable...")

    model = PeftModel.from_pretrained(
        model,
        V5B_MODEL,
        is_trainable=True,
    )

else:

    print()
    print("V5B appears to be a merged/full model.")

    print("Creating new V5C LoRA adapter...")

    model = get_peft_model(
        model,
        lora_config,
    )


# ============================================================
# 15. TRAINABLE PARAMETERS
# ============================================================

print("\n" + "=" * 100)
print("TRAINABLE PARAMETERS")
print("=" * 100)

model.print_trainable_parameters()


# ============================================================
# 16. LOAD ONLY V5C DATASET
# ============================================================

print("\n" + "=" * 100)
print("LOADING V5C DATASET")
print("=" * 100)

dataset = load_dataset(
    "json",
    data_files=V5C_DATASET,
    split="train",
)

print(dataset)

print()
print("Number of V5C samples:", len(dataset))


# ============================================================
# 17. CONVERT V5C DATASET
# ============================================================
#
# Your dataset:
#
# {
#   "instruction": "Summarize the conversation",
#   "input": {
#       "relationship": "Friend",
#       "conversation": "..."
#   },
#   "output": {
#       "summary": "..."
#   }
# }
#
# becomes:
#
# system
# user
# assistant
#
# ============================================================


SYSTEM_PROMPT = """You are Compatifi V5C.

Your task is to summarize a relationship conversation.

Generate ONE concise factual summary.

Include only information supported by the conversation.

Focus on:
- the main topic
- important facts
- the user's situation or goal when relevant
- important issue or outcome

Do not:
- generate a reply
- give advice
- extract memories
- invent information
- add opinions
- include unnecessary details

For conversations with very little information, summarize whatever factual information is actually present.

Return only the summary.
"""


def build_messages(example):

    input_data = example["input"]
    output_data = example["output"]

    relationship = str(
        input_data["relationship"]
    ).strip()

    conversation = str(
        input_data["conversation"]
    ).strip()

    summary = str(
        output_data["summary"]
    ).strip()

    user_prompt = (
        "Relationship: "
        + relationship
        + "\n\n"
        + "Conversation:\n"
        + conversation
        + "\n\n"
        + "Task: Summarize the conversation."
    )

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

        {
            "role": "user",
            "content": user_prompt,
        },

        {
            "role": "assistant",
            "content": summary,
        },

    ]

    return messages


# ============================================================
# 18. FORMAT DATASET
# ============================================================

def format_example(example):

    messages = build_messages(example)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text
    }


print("\nFormatting V5C dataset...")

formatted_dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
)

print("✅ Dataset formatted.")


# ============================================================
# 19. TOKENIZATION WITH ASSISTANT-ONLY LOSS
# ============================================================
#
# This is important.
#
# We want V5C to learn:
#
#     conversation -> summary
#
# rather than wasting loss on reproducing the prompt.
#
# Therefore:
#
# prompt tokens = label -100
# summary tokens = actual labels
#
# ============================================================


def tokenize_example(example):

    messages = build_messages(
        {
            "input": {
                "relationship": "",
                "conversation": "",
            },
            "output": {
                "summary": "",
            },
        }
    )

    # --------------------------------------------------------
    # We reconstruct the actual conversation messages
    # from the already formatted text indirectly below.
    # --------------------------------------------------------

    full_text = example["text"]

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # --------------------------------------------------------
    # Find assistant response boundary.
    #
    # We create the assistant portion separately.
    # --------------------------------------------------------

    # The formatted text is based on:
    #
    # system + user + assistant
    #
    # We need the assistant summary tokens.

    # Get assistant messages by parsing the original text
    # is unreliable, so this function is replaced below.
    #
    # Kept here only to avoid accidental use.
    # --------------------------------------------------------

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
    }


# ============================================================
# 20. BETTER TOKENIZATION
# ============================================================
#
# Rebuild each example directly so we can mask the prompt.
# ============================================================


def tokenize_v5c(example):

    relationship = str(
        example["input"]["relationship"]
    ).strip()

    conversation = str(
        example["input"]["conversation"]
    ).strip()

    summary = str(
        example["output"]["summary"]
    ).strip()

    user_prompt = (
        "Relationship: "
        + relationship
        + "\n\n"
        + "Conversation:\n"
        + conversation
        + "\n\n"
        + "Task: Summarize the conversation."
    )

    # --------------------------------------------------------
    # Prompt WITHOUT assistant answer
    # --------------------------------------------------------

    prompt_messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

        {
            "role": "user",
            "content": user_prompt,
        },

    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Full training example
    # --------------------------------------------------------

    full_messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

        {
            "role": "user",
            "content": user_prompt,
        },

        {
            "role": "assistant",
            "content": summary,
        },

    ]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )

    # --------------------------------------------------------
    # Tokenize complete example
    # --------------------------------------------------------

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]

    attention_mask = full_tokens["attention_mask"]

    prompt_length = len(
        prompt_tokens["input_ids"]
    )

    # --------------------------------------------------------
    # Create labels
    #
    # - Prompt = -100
    # - Assistant summary = actual token IDs
    # --------------------------------------------------------

    labels = input_ids.copy()

    prompt_length = min(
        prompt_length,
        len(labels)
    )

    for i in range(prompt_length):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("\n" + "=" * 100)
print("TOKENIZING V5C")
print("=" * 100)

tokenized_dataset = dataset.map(
    tokenize_v5c,
    remove_columns=dataset.column_names,
)

print("✅ V5C tokenization complete.")


# ============================================================
# 21. CUSTOM DATA COLLATOR
# ============================================================

class V5CDataCollator:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, features):

        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_token_id = self.tokenizer.pad_token_id

        for feature in features:

            input_ids = feature["input_ids"]

            attention_mask = feature["attention_mask"]

            labels = feature["labels"]

            padding_length = (
                max_length
                - len(input_ids)
            )

            input_ids = (
                input_ids
                + [pad_token_id] * padding_length
            )

            attention_mask = (
                attention_mask
                + [0] * padding_length
            )

            labels = (
                labels
                + [-100] * padding_length
            )

            batch_input_ids.append(
                input_ids
            )

            batch_attention_mask.append(
                attention_mask
            )

            batch_labels.append(
                labels
            )

        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
        }


data_collator = V5CDataCollator(
    tokenizer
)


# ============================================================
# 22. TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 100)
print("TRAINING CONFIGURATION")
print("=" * 100)

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    warmup_ratio=WARMUP_RATIO,

    lr_scheduler_type="cosine",

    fp16=True,

    bf16=False,

    optim="paged_adamw_8bit",

    logging_steps=LOGGING_STEPS,

    logging_first_step=True,

    save_strategy="steps",

    save_steps=SAVE_STEPS,

    save_total_limit=SAVE_TOTAL_LIMIT,

    gradient_checkpointing=True,

    remove_unused_columns=False,

    dataloader_pin_memory=True,

    report_to="none",

    save_safetensors=True,

)


# ============================================================
# 23. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator,
)


# ============================================================
# 24. FIND LATEST CHECKPOINT
# ============================================================

print("\n" + "=" * 100)
print("CHECKING FOR EXISTING V5C CHECKPOINT")
print("=" * 100)

latest_checkpoint = None

if os.path.isdir(OUTPUT_DIR):

    latest_checkpoint = get_last_checkpoint(
        OUTPUT_DIR
    )


# ============================================================
# 25. START / RESUME TRAINING
# ============================================================

if latest_checkpoint:

    print()
    print("✅ CHECKPOINT FOUND")

    print()
    print(
        "Latest checkpoint:"
    )

    print(
        os.path.abspath(
            latest_checkpoint
        )
    )

    print()
    print(
        "Resuming V5C training..."
    )

    trainer.train(
        resume_from_checkpoint=latest_checkpoint
    )

else:

    print()
    print("ℹ️ NO V5C CHECKPOINT FOUND")

    print()
    print(
        "Starting new V5C training from V5B..."
    )

    trainer.train()


# ============================================================
# 26. SAVE FINAL V5C
# ============================================================

print("\n" + "=" * 100)
print("SAVING V5C")
print("=" * 100)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 100)
print("✅ V5C TRAINING COMPLETE")
print("=" * 100)

print()
print("V5C saved to:")

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print()
print("Training source:")
print("V5B")

print()
print("Training dataset:")
print("V5C ONLY")

print()
print("Replay dataset:")
print("NONE")

print()
print("Previous-stage dataset:")
print("NONE")

print()
print("V5C task:")
print("Conversation -> concise factual summary")

print("=" * 100)