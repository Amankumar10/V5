import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "../V4B_Final_Merged_Model"

print("=" * 70)
print("V5A MODEL LOAD TEST")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print("Compute dtype:", compute_dtype)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully.")

print("\nConfiguring 4-bit QLoRA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
print("This can take a few minutes.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
)

print("\n" + "=" * 70)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 70)

print("Model:", MODEL_NAME)
print("Model type:", model.config.model_type)
print("Parameters:", sum(p.numel() for p in model.parameters()))

print("\nGPU memory:")
print(
    f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)
print(
    f"Reserved : {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
)

print("=" * 70)