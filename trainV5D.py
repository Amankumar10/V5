# Compatifi V5D TRL Training
# V5C merged model -> V5D dataset -> V5D LoRA adapter

import json, os, sys
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, trainer_utils

BASE_CHECKPOINT = "./V5C_Final_Merged_Model"
V5D_DATASET = r".\datasets\V5D\V5D.jsonl"
NORMALIZED_DATASET = r".\v5d_normalized_training.jsonl"
OUTPUT_DIR = r".\V5D_Final"

MAX_SEQ_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 2
WARMUP_RATIO = 0.05
SAVE_STEPS = 150
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2

SYSTEM_PROMPT = """You are Compatifi V5D.

Your ONLY task is to extract structured understanding of the current relationship conversation.

V5D answers: "What is happening in this conversation?"

Identify:
- the main topic
- the user's current situation
- the user's current goal
- the current issue or obstacle
- the conversation stage
- whether the user needs help
- the type of help required
- any relevant risk
- overall confidence

Do not generate replies, advice, coaching, compatibility analysis, personality analysis,
long-term memories, profiles, or unsupported information.

The conversation summary is supporting context. The conversation itself is primary evidence.

Unknown string fields must be "".
help_required must always be true or false.
risk_detection fields must always be true or false.
confidence must be between 0 and 1.

Return only valid JSON."""

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU was not detected.")

if not os.path.exists(BASE_CHECKPOINT):
    raise FileNotFoundError(os.path.abspath(BASE_CHECKPOINT))
if not os.path.isfile(V5D_DATASET):
    raise FileNotFoundError(os.path.abspath(V5D_DATASET))

print("=" * 80)
print("COMPATIFI V5D TRAINING")
print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))

USE_BF16 = torch.cuda.is_bf16_supported()
USE_FP16 = not USE_BF16
compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_CHECKPOINT,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

records = []
with open(V5D_DATASET, "r", encoding="utf-8") as f:
    for n, line in enumerate(f, 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not all(k in item for k in ("instruction","input","output")):
            raise ValueError(f"Missing required keys at line {n}")
        records.append(item)

if not records:
    raise ValueError("V5D dataset is empty.")

def dump(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))

formatted = []
for item in records:
    user_content = f"Instruction:\n{item['instruction']}\n\nInput:\n{dump(item['input'])}"
    assistant_content = dump(item["output"])
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":user_content},
        {"role":"assistant","content":assistant_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        text = f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{user_content}\n\n### Assistant:\n{assistant_content}"
    formatted.append({"text": text})

formatted_dataset = Dataset.from_list(formatted)

with open(NORMALIZED_DATASET, "w", encoding="utf-8") as f:
    for row in formatted:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("V5D samples:", len(formatted_dataset))

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    optim="paged_adamw_8bit",
    bf16=USE_BF16,
    fp16=USE_FP16,
    tf32=True,
    gradient_checkpointing=True,
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to="none",
    disable_tqdm=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.model.print_trainable_parameters()

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = trainer_utils.get_last_checkpoint(OUTPUT_DIR)

print("Checkpoint:", last_checkpoint or "None")
trainer.train(resume_from_checkpoint=last_checkpoint)

FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("=" * 80)
print("V5D TRAINING COMPLETE")
print("Final model:", os.path.abspath(FINAL_MODEL_DIR))
print("=" * 80)
