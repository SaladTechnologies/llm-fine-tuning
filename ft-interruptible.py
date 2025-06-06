#import os
#import re
#import math
#from tqdm import tqdm
#from huggingface_hub import login
from datetime import datetime, timezone

import torch
# In PyTorch 2.6 and later, the default behavior of torch.load() changed — weights_only=True is now the default.
# This monkey patch restores the full checkpoint loading behavior from PyTorch 2.5, allowing deserialization of the entire training state (model, optimizer, Trainer state, etc.).
_original_torch_load = torch.load
def unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = unsafe_torch_load

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig            
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
#from datetime import datetime

###################### For running on SaladCloud - 1st Change: Get the parameters, start the uploader thread, filter node, and sync data from cloud to local
import copy
from helper import  Resume_From_Cloud, Get_Checkpoint, Notify_Uploader, Close_All, g_TASK_NAME, g_SEED, g_MODEL, g_EPOCHS, g_BATCH_SIZE, g_SAVING_STEPS
Resume_From_Cloud()  
######################

# Custom callback for printing state information when (after) checkpoints are saved
class CheckpointCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        ##################### For running on SaladCloud - 3rd Change: Notify the uploader thread to upload the latest checkpoint
        Notify_Uploader( copy.deepcopy (state.__dict__) ) # Ensuring the uploader thread receives a frozen snapshot of the state
        #####################    
        #print(60 * "*" + " The current training state")
        #print(state)
        #print(60 * "*") 
        return control

# Set a global seed to ensure the dataset shuffle order is deterministic across runs and epochs.
# Each epoch receives a consistent shuffle based on the combination of the seed and the epoch number.
# Hugging Face's Trainer can save checkpoints during training, enabling a seamless resume from where it left off, including the model state, optimizer, gradients, and the position within the shuffled dataset for the epoch.
set_seed(g_SEED) 

# Model setup
BASE_MODEL = g_MODEL 

MAX_SEQUENCE_LENGTH = 182
PROJECT_RUN_NAME = g_TASK_NAME

# LoRA adpater configuration
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.1
QUANT_4_BIT = True

# Training hyperparameters
EPOCHS = g_EPOCHS
BATCH_SIZE = g_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = 1 
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = 'cosine'
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"

# Logging and saving config
STEPS      = g_SAVING_STEPS   # logging
SAVE_STEPS = g_SAVING_STEPS   # saving
LOG_TO_WANDB = False

# Load dataset from Hugging Face Hub
print("Downloading the dataset at " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), flush=True)
dataset = load_dataset("ed-donner/pricer-data")
train = dataset['train']  # Original size: 400,000 samples
test = dataset['test']    # Original size:   2,000 samples

# Select a subset of samples for faster experimentation
train = train.select(range(400000)) 
#test = test.select(range(2000))

# Print a sample
#print(60 * "*" + " A sample")
#print(train[0])

# BitsAndBytesConfig for quantization
if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
  )

# Load tokenizer and adjust padding config
print("Downloading the tokenizer at " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Padding for batched traning
tokenizer.padding_side = "right"

# Load the quantized base model
print("Downloading the model at " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
#print(60 * "*" + " The memory footprint")
#print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

# Data collator with loss masking: Only calculate loss after "Price is $" in target text, guiding the model to learn how to generate the correct completion rather than memorizing the prompt.
response_template = "Price is $"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# LoRA configuration (added on top of frozen base model layers)
# THe LoRA adapters are being trained on top of the 4-bit quantized base model, but they are not quantized.
lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

# Define training arguments
train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,                              # Where to save checkpoints
    num_train_epochs=EPOCHS,                                  # Number of full passes over the training dataset.
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="no",                                       # No evaluation during training
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Instead of updating the model weights every batch, it waits for N batches, accumulates gradients, and then updates once.
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,                                      # Keep only the last 10 checkpoints to save disk space. Older ones will be deleted.
    logging_steps=STEPS,                                      # Frequency of logging
    disable_tqdm=True,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,                                             # Use epochs instead
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,                                     # Groups sequences of similar lengths into the same batch. Reduces padding, improving training efficiency.
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to = "none", 
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dataset_text_field="text",
    include_num_input_tokens_seen = True,
    save_strategy="steps"                                     # "no", "epoch", "steps"
)

# Initialize trainer with model, dataset, tokenizer, collator, LoRA, and callback
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    peft_config=lora_parameters,
    tokenizer=tokenizer,
    args=train_parameters,
    data_collator=collator,
    callbacks=[CheckpointCallback()] 
)

# Train the model
print("Training started at " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), flush=True)
##################### For running on SaladCloud - 2nd Change: Resume training from the latest checkpoint 
Notify_Uploader('start') # Optional: Signal the uploader thread that the model downloaded and training started
temp = Get_Checkpoint()
if temp == "":
    fine_tuning.train()
else:
    fine_tuning.train(resume_from_checkpoint=PROJECT_RUN_NAME + "/" + temp) 
#####################

# Save final model
fine_tuning.save_model(output_dir=PROJECT_RUN_NAME + "/final")
print("Training completed at " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), flush=True)

##################### For running on SaladCloud - 4th Change:  Wait for all checkpoints and the final model to finish uploading , and then shutdown the container group
Close_All()
#####################