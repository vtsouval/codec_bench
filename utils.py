import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from datasets import DatasetDict

def get_subset(dataset, subset_size=None, subset_fraction=0.1, seed=42):
	assert 'train' in dataset and 'val' in dataset and 'test' in dataset, "Dataset must contain train, val, and test splits"
	random.seed(seed)
	subset = {}
	for split in ['train', 'val', 'test']:
		ds_split = dataset[split]
		n = len(ds_split)

		if subset_size is not None:
			size = min(subset_size, n)
		elif subset_fraction is not None:
			size = max(1, int(n * subset_fraction))
		else:
			raise ValueError("Either subset_size or subset_fraction must be provided.")
		indices = random.sample(range(n), size)
		subset[split] = ds_split.select(indices)
	return DatasetDict(subset)

def load_tokenizer(model="HKUSTAudio/Llasa-1B-Multilingual", max_length=128):
	tokenizer = AutoTokenizer.from_pretrained(model, model_max_length=max_length, padding_side="right", use_fast=True, local_files_only=True,)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.text_vocab_size = 128000
	tokenizer.speech_vocab_size = 65536
	tokenizer.reserved_tokens = 263
	tokenizer.vocab_size_override = 193800
	return tokenizer

def load_model(model="HKUSTAudio/Llasa-1B-Multilingual", device="auto", grad_ckpt=True, quant_cfg=None):
	model =  AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16 if quant_cfg is None else None,
		device_map=device, attn_implementation="flash_attention_2", use_cache=False, quantization_config=quant_cfg)
	if grad_ckpt:
		model.gradient_checkpointing_enable()
	return model

def load_peft_config(lora_r=8, lora_alpha=32, lora_modules=["q_proj", "v_proj", "gate_proj", "up_proj"], lora_dropout=0.1):
	return LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")

def load_bnb_config():
	return BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_use_double_quant=True, bnb_8bit_quant_type="nf4", bnb_8bit_compute_dtype=torch.bfloat16)

def preprocess_logits_for_metrics(logits, labels):
	if isinstance(logits, tuple):
		logits = logits[0]
	return logits.argmax(dim=-1)

def compute_accuracy(eval_preds, tokenizer):
	preds, labels = eval_preds
	mask = labels != -100
	preds, labels = preds[mask], labels[mask]
	# Decode token IDs to token strings
	labels_text = tokenizer.convert_ids_to_tokens(labels, skip_special_tokens=True)
	preds_text = tokenizer.convert_ids_to_tokens(preds, skip_special_tokens=True)
	accuracy = np.mean(np.array(preds_text) == np.array(labels_text)) if len(labels_text) > 0 else 0.0
	print("Examples: (label/pred): ", list(zip(labels_text[-3:], preds_text[-3:])))
	return {"accuracy": accuracy}

class EarlyStopper:
	def __init__(self, patience=3):
		self.patience = patience
		self.counter = 0
		self.best_acc = 0.0
		self.best_loss = float("inf")

	def should_stop(self, eval_metrics):
		acc = eval_metrics["acc"]
		loss = eval_metrics["loss"]
		acc_improved = acc > self.best_acc
		loss_improved = loss < self.best_loss
		if acc >= 1.0 or loss <= 0.0:
			if acc==1.0:
				print("Early stopping: perfect accuracy reached.")
			else:
				print("Early stopping: zero loss reached.")
			return True
		if acc_improved or loss_improved:
			if acc_improved:
				self.best_acc = acc
			if loss_improved:
				self.best_loss = loss
			self.counter = 0
		else:
			self.counter += 1
		if self.counter >= self.patience:
			print("Early stopping triggered due to no improvement.")
			return True
		return False

if __name__ == "__main__":
	model = load_model()
	print(model)