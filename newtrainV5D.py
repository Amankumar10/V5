# =============================================================================
# COMPATIFI V5D — BALANCED CONTINUAL LEARNING
# NVIDIA L40S 48GB OPTIMIZED
#
# Pipeline:
# V5A → V5B → V5C → V5C Final Merged Base → V5D QLoRA → V5D Adapter
#
# Goal:
# Learn V5D Conversation Understanding while minimizing drift from
# previously acquired V5A/V5B/V5C capabilities.
# =============================================================================

import os
import json
import random

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_CHECKPOINT = "./V5C_Final_Merged_Model"
V5D_DATASET = "./datasets/V5D/V5D.jsonl"
OUTPUT_DIR = "./V5D_Final"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
NORMALIZED_DATASET = os.path.join(
    OUTPUT_DIR,
    "v5d_normalized_training.jsonl",
)

# Reproducibility
SEED = 42

# L40S / Sequence Configuration
MAX_SEQ_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = (
    PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

# Balanced Continual Learning
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# Validation
VALIDATION_SPLIT = 0.05

# Logging / Checkpoints
SAVE_STEPS = 500
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant.

Analyze the provided conversation according to the instruction.

Use only information supported by the provided context.

Return the requested output accurately and concisely."""


# =============================================================================
# REQUIRED V5D OUTPUT SCHEMA
# =============================================================================

REQUIRED_OUTPUT_FIELDS = {
    "topic",
    "user_situation",
    "current_goal",
    "current_issue",
    "conversation_stage",
    "help_required",
    "help_type",
    "risk_detection",
    "confidence",
}

REQUIRED_RISK_FIELDS = {
    "scam",
    "threat",
    "harassment",
    "manipulation",
}


# =============================================================================
# UTILITIES
# =============================================================================

def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_json(value):
    """Serialize non-string values into compact JSON."""
    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_record(item, line_number):
    """Validate one V5D dataset record."""

    # Top-level validation
    if not isinstance(item, dict):
        raise ValueError(
            f"Line {line_number}: Record must be a JSON object."
        )

    required_keys = {"instruction", "input", "output"}
    missing = required_keys - set(item.keys())

    if missing:
        raise ValueError(
            f"Line {line_number}: Missing keys: {missing}"
        )

    # Input validation
    input_data = item["input"]

    if not isinstance(input_data, dict):
        raise ValueError(
            f"Line {line_number}: input must be an object."
        )

    required_input = {
        "relationship",
        "conversation",
        "conversation_summary",
    }

    missing_input = required_input - set(input_data.keys())

    if missing_input:
        raise ValueError(
            f"Line {line_number}: "
            f"Missing input fields: {missing_input}"
        )

    # Output validation
    output = item["output"]

    if not isinstance(output, dict):
        raise ValueError(
            f"Line {line_number}: output must be an object."
        )

    missing_output = REQUIRED_OUTPUT_FIELDS - set(output.keys())

    if missing_output:
        raise ValueError(
            f"Line {line_number}: "
            f"Missing output fields: {missing_output}"
        )

    # help_required validation
    if not isinstance(output["help_required"], bool):
        raise ValueError(
            f"Line {line_number}: help_required must be boolean."
        )

    # Risk detection validation
    risk = output["risk_detection"]

    if not isinstance(risk, dict):
        raise ValueError(
            f"Line {line_number}: "
            f"risk_detection must be an object."
        )

    missing_risk = REQUIRED_RISK_FIELDS - set(risk.keys())

    if missing_risk:
        raise ValueError(
            f"Line {line_number}: "
            f"Missing risk fields: {missing_risk}"
        )

    for field in REQUIRED_RISK_FIELDS:
        if not isinstance(risk[field], bool):
            raise ValueError(
                f"Line {line_number}: "
                f"risk_detection.{field} must be boolean."
            )

    # Confidence validation
    confidence = output["confidence"]

    if not isinstance(confidence, (int, float)):
        raise ValueError(
            f"Line {line_number}: confidence must be numeric."
        )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Line {line_number}: "
            f"confidence must be between 0 and 1."
        )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_records(dataset_path):
    """Load and validate JSONL dataset."""

    print_section("LOADING AND VALIDATING V5D DATASET")

    records = []

    with open(
        dataset_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {error}"
                )

            validate_record(item, line_number)
            records.append(item)

    if not records:
        raise ValueError("V5D dataset is empty.")

    print(f"Validated V5D samples: {len(records)}")

    if len(records) < 10_000:
        print("Dataset size: Small")
    elif len(records) < 60_000:
        print("Dataset size: Medium")
    else:
        print("Dataset size: Large")

    return records


# =============================================================================
# TOKENIZATION
# =============================================================================

def build_tokenized_dataset(records, tokenizer):
    """
    Build tokenized records with assistant-only loss.

    Prompt tokens → label = -100
    Assistant tokens → label = actual token ID
    """

    print_section("TOKENIZING DATASET")

    tokenized_records = []
    normalized_records = []

    for index, item in enumerate(records, start=1):

        # ---------------------------------------------------------------------
        # User Content
        # ---------------------------------------------------------------------

        user_content = (
            f"Instruction:\n{item['instruction']}\n\n"
            f"Input:\n{dump_json(item['input'])}"
        )

        assistant_content = dump_json(item["output"])

        # ---------------------------------------------------------------------
        # Chat Messages
        # ---------------------------------------------------------------------

        prompt_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        full_messages = [
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

        # ---------------------------------------------------------------------
        # Apply Model Chat Template
        # ---------------------------------------------------------------------

        if getattr(tokenizer, "chat_template", None):

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        else:
            prompt_text = (
                f"### System:\n"
                f"{SYSTEM_PROMPT}\n\n"
                f"### User:\n"
                f"{user_content}\n\n"
                f"### Assistant:\n"
            )

            full_text = prompt_text + assistant_content

        # ---------------------------------------------------------------------
        # Tokenization
        # ---------------------------------------------------------------------

        full_encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False,
        )

        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False,
        )

        input_ids = full_encoding["input_ids"]

        # ---------------------------------------------------------------------
        # Assistant-Only Labels
        # ---------------------------------------------------------------------

        prompt_length = min(
            len(prompt_encoding["input_ids"]),
            len(input_ids),
        )

        labels = input_ids.copy()

        for token_index in range(prompt_length):
            labels[token_index] = -100

        assistant_tokens = sum(
            label != -100
            for label in labels
        )

        # Skip samples where truncation removed almost all output
        if assistant_tokens < 5:
            continue

        # ---------------------------------------------------------------------
        # Save Tokenized Record
        # ---------------------------------------------------------------------

        tokenized_records.append({
            "input_ids": input_ids,
            "attention_mask": full_encoding["attention_mask"],
            "labels": labels,
        })

        normalized_records.append({
            "prompt": prompt_text,
            "completion": assistant_content,
        })

        if index % 10_000 == 0:
            print(f"Processed {index:,} samples...")

    if not tokenized_records:
        raise ValueError(
            "No valid tokenized samples remain."
        )

    print(
        f"Final training samples: "
        f"{len(tokenized_records):,}"
    )

    return tokenized_records, normalized_records


# =============================================================================
# SAVE NORMALIZED DATASET
# =============================================================================

def save_normalized_dataset(records, output_path):
    """Save readable prompt/completion dataset."""

    print("\nSaving normalized dataset...")

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "Normalized dataset saved:"
    )
    print(os.path.abspath(output_path))


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================

    set_seed(SEED)

    # =========================================================================
    # CUDA VALIDATION
    # =========================================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected."
        )

    print_section("COMPATIFI V5D TRAINING")
    print("Balanced Continual Learning")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_properties = torch.cuda.get_device_properties(0)
    gpu_memory_gb = (
        gpu_properties.total_memory / 1024 ** 3
    )

    print(f"GPU:  {gpu_name}")
    print(f"VRAM: {gpu_memory_gb:.2f} GB")

    # =========================================================================
    # PATH VALIDATION
    # =========================================================================

    if not os.path.exists(BASE_CHECKPOINT):
        raise FileNotFoundError(
            f"\nBase checkpoint not found:\n"
            f"{os.path.abspath(BASE_CHECKPOINT)}"
        )

    if not os.path.isfile(V5D_DATASET):
        raise FileNotFoundError(
            f"\nV5D dataset not found:\n"
            f"{os.path.abspath(V5D_DATASET)}"
        )

    print("\nBase Model:")
    print(os.path.abspath(BASE_CHECKPOINT))

    print("\nDataset:")
    print(os.path.abspath(V5D_DATASET))

    # =========================================================================
    # PRECISION
    # =========================================================================

    USE_BF16 = torch.cuda.is_bf16_supported()

    if not USE_BF16:
        raise RuntimeError(
            "BF16 support was expected but not detected."
        )

    compute_dtype = torch.bfloat16
    print("\nPrecision: BF16")

    # =========================================================================
    # LOAD TOKENIZER
    # =========================================================================

    print_section("LOADING TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_CHECKPOINT,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Tokenizer loaded.")

    # =========================================================================
    # QLORA CONFIGURATION
    # =========================================================================

    print_section("CONFIGURING QLORA")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # =========================================================================
    # LOAD V5C MERGED BASE MODEL
    # =========================================================================

    print_section("LOADING V5C FINAL MERGED MODEL")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_CHECKPOINT,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    print("V5C merged model loaded.")

    # =========================================================================
    # PREPARE FOR QLORA
    # =========================================================================

    print("\nPreparing model for QLoRA training...")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # =========================================================================
    # LORA CONFIGURATION
    # =========================================================================

    print_section("CONFIGURING BALANCED LORA")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",

        # Attention + MLP adaptation
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

    model = get_peft_model(
        model,
        peft_config,
    )

    print("Balanced LoRA configured.")

    # =========================================================================
    # LOAD DATASET
    # =========================================================================

    records = load_records(V5D_DATASET)

    # =========================================================================
    # TOKENIZE DATASET
    # =========================================================================

    tokenized_records, normalized_records = (
        build_tokenized_dataset(
            records,
            tokenizer,
        )
    )

    # =========================================================================
    # SAVE NORMALIZED DATASET
    # =========================================================================

    save_normalized_dataset(
        normalized_records,
        NORMALIZED_DATASET,
    )

    # =========================================================================
    # CREATE HF DATASET
    # =========================================================================

    full_dataset = Dataset.from_list(
        tokenized_records
    )

    # =========================================================================
    # TRAIN / VALIDATION SPLIT
    # =========================================================================

    print_section(
        "CREATING TRAIN / VALIDATION SPLIT"
    )

    dataset_split = full_dataset.train_test_split(
        test_size=VALIDATION_SPLIT,
        seed=SEED,
        shuffle=True,
    )

    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(
        f"Training samples:   {len(train_dataset):,}"
    )
    print(
        f"Validation samples: {len(eval_dataset):,}"
    )

    # =========================================================================
    # DATA COLLATOR
    # =========================================================================

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # =========================================================================
    # TRAINING ARGUMENTS
    # =========================================================================

    print_section("CONFIGURING TRAINING")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Learning
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        # Optimizer
        optim="paged_adamw_8bit",

        # Precision
        bf16=True,
        fp16=False,
        tf32=True,

        # Memory
        gradient_checkpointing=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,

        # Checkpointing
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        # Best Model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=LOGGING_STEPS,
        report_to="none",
        disable_tqdm=False,

        # Dataset
        remove_unused_columns=False,

        # Reproducibility
        seed=SEED,
    )

    # =========================================================================
    # TRAINER
    # =========================================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ],
    )

    # =========================================================================
    # TRAINING SUMMARY
    # =========================================================================

    print_section(
        "BALANCED CONTINUAL LEARNING CONFIGURATION"
    )

    trainer.model.print_trainable_parameters()

    print(f"\nBase Model: V5C Final Merged Model")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Effective Batch Size: {EFFECTIVE_BATCH_SIZE}")
    print(f"LoRA Rank: {peft_config.r}")
    print(f"LoRA Alpha: {peft_config.lora_alpha}")
    print("LoRA Targets: Attention + MLP projections")
    print("Assistant-only loss: Enabled")
    print(f"Validation Split: {VALIDATION_SPLIT * 100}%")
    print("Best Checkpoint Restoration: Enabled")
    print("Early Stopping: Enabled")

    # =========================================================================
    # CHECKPOINT DETECTION
    # =========================================================================

    last_checkpoint = None

    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(
            OUTPUT_DIR
        )

    print(
        "\nPrevious checkpoint:",
        last_checkpoint or "None",
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print_section("STARTING V5D TRAINING")

    trainer.train(
        resume_from_checkpoint=last_checkpoint
    )

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================

    print_section("FINAL VALIDATION")

    final_metrics = trainer.evaluate()

    print("\nFinal evaluation results:")

    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    # =========================================================================
    # SAVE FINAL ADAPTER
    # =========================================================================

    print_section("SAVING FINAL V5D ADAPTER")

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

    print("\nFinal V5D adapter saved:")
    print(
        os.path.abspath(
            FINAL_MODEL_DIR
        )
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print_section("V5D TRAINING COMPLETE")

    print("""
Pipeline:
V5A Knowledge
    ↓
V5B Knowledge
    ↓
V5C Knowledge
    ↓
V5C Final Merged Model
    ↓
Balanced V5D QLoRA
    ↓
V5D Final Adapter

Training Design:
✓ Previous V5C merged model as frozen base
✓ 4-bit NF4 QLoRA
✓ BF16 training
✓ Learning rate = 2e-5
✓ 1 epoch
✓ LoRA rank = 16
✓ LoRA alpha = 32
✓ Attention + MLP LoRA
✓ Assistant-only loss
✓ Validation monitoring
✓ Best checkpoint restoration
✓ Early stopping
✓ Gradient checkpointing

Goal:
Learn V5D Conversation Understanding strongly while minimizing
unnecessary drift from V5A/V5B/V5C capabilities.
""")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()