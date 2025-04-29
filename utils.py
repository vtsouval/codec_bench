import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(model="HKUSTAudio/Llasa-1B-Multilingual"):
	tokenizer = AutoTokenizer.from_pretrained(
		model,
		model_max_length=2048,
		padding_side="right",
		use_fast=True,
		local_files_only=True,
	)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.text_vocab_size = 128000
	tokenizer.speech_vocab_size = 65536
	tokenizer.reserved_tokens = 263
	return tokenizer

def load_model(model="HKUSTAudio/Llasa-1B-Multilingual"):
	model = AutoModelForCausalLM.from_pretrained(
		model,
		torch_dtype=torch.bfloat16,
		device_map="cuda:0",
		attn_implementation="flash_attention_2"
	)
	return model

def compute_accuracy(eval_preds):
	import re
	_filter = lambda x: int(re.search(r"\d+", x).group(0))
	logits, labels = eval_preds
	labels = torch.tensor(labels)
	predictions = torch.tensor(logits).argmax(dim=-1)
	valid_masks = labels != -100
	correct, total = 0, 0
	for label_ids, pred_ids, mask in zip(labels, predictions, valid_masks):
		label_tokens, pred_tokens = label_ids[mask], pred_ids[mask]
		true_label = _filter(tokenizer.decode(label_tokens.tolist(), skip_special_tokens=True))
		pred_label = _filter(tokenizer.decode(pred_tokens.tolist(), skip_special_tokens=True))
		correct += int(true_label == pred_label)
		total += 1
	accuracy = correct / total if total > 0 else 0.0
	return {"accuracy": accuracy}
