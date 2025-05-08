import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM
try:
	from ..utils import load_tokenizer
except ImportError:
	from utils import load_tokenizer

DEVICE = "cuda:0"
MODEL_PATH = "/home/isma/GitHub/codec_bench/assets/audiomnist_Llasa-1B_xcodec2_True/checkpoint-40"
DS_DIR = "./datasets"
DS_NAME = "audiomnist"

def main():
	model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
	model.eval()
	tokenizer = load_tokenizer()
	data = load_from_disk(f"{DS_DIR}/{DS_NAME}_tok")
	test_data = data["test"]
	test_data.set_format(type="torch")

	for sample in test_data:
		input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
		attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
		mask = sample["labels"] != -100
		label_ids = sample["labels"][mask]
		with torch.no_grad():
			pred_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)[0].argmax(dim=-1).squeeze(0)
		pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
		label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
		print(f"Preds: {pred_text}, True: {label_text} ({label_ids})")

if __name__ == "__main__":
    main()