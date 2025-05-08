import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import load_from_disk
from trainer import CustomTrainer
from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback
from utils import load_tokenizer, load_model
import argparse

parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument("--save_dir", type=str, default="./assets", help="Directory to save models and outputs")
parser.add_argument("--ds_dir", type=str, default="./datasets", help="Path to the datasets directory")
parser.add_argument("--ds_name", type=str, default="esc50", help="Dataset name")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size to be used in training")
parser.add_argument("--model_name", type=str, default="Llasa-1B", help="Model name")
parser.add_argument("--codec_name", type=str, default="xcodec2", help="Codec name")
parser.add_argument("--override", action="store_true", help="Whether to override existing outputs")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--eval_freq", type=int, default=100, help="Evaluation frequency in steps")
parser.add_argument("--patience", type=int, default=30, help="Patience for early stopping")
args = parser.parse_args()

if __name__ == "__main__":

	########################################################################################################################################################################
	print("Loading tokenizer...")
	tokenizer = load_tokenizer(max_length=128)
	########################################################################################################################################################################

	########################################################################################################################################################################
	print(f"Loading dataset {args.ds_name}...")
	data = load_from_disk(f"{args.ds_dir}/{args.ds_name}")
	data.set_format(type="torch")
	train_ds = torch.utils.data.DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, collate_fn=DataCollatorForTokenClassification(tokenizer, padding="longest"), pin_memory=True, num_workers=4)
	eval_ds = torch.utils.data.DataLoader(data["test"], batch_size=args.batch_size, shuffle=False, collate_fn=DataCollatorForTokenClassification(tokenizer, padding="longest"), pin_memory=True, num_workers=4)
	########################################################################################################################################################################

	########################################################################################################################################################################
	print(f"Loading model {args.model_name}...")
	model = load_model(grad_ckpt=False)
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
	########################################################################################################################################################################

	########################################################################################################################################################################
	trainer = CustomTrainer(output_dir=f"{args.save_dir}/{args.ds_name}_{args.model_name}_{args.codec_name}", eval_steps=args.eval_freq, num_train_epochs=args.num_epochs,
		overwrite_output_dir=args.override, model=model, tokenizer=tokenizer, optimizer=optimizer, lr_scheduler_type="cosine",
		max_grad_norm=1.0, callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
	trainer.train(train_ds=train_ds, eval_ds=eval_ds)
	########################################################################################################################################################################



