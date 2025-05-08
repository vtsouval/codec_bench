import os
import json
import shutil
import torch
import numpy as np
from tqdm import tqdm
from transformers import EarlyStoppingCallback, PreTrainedModel, PreTrainedTokenizer, TrainerState, TrainerControl, get_scheduler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import f1_score

class CustomArgs:
	def __init__(self, metric_for_best_model="eval_acc", greater_is_better=False, load_best_model_at_end=True):
		self.metric_for_best_model = metric_for_best_model
		self.greater_is_better = greater_is_better
		self.load_best_model_at_end = load_best_model_at_end
		self.eval_strategy = "steps"

class CustomTrainer:

	def __init__(
		self,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizer,
		optimizer: torch.optim.Optimizer,
		lr_scheduler_type: str = "cosine",
		max_grad_norm: float = 1.0,
		num_train_epochs: int = 5,
		eval_steps: int = 100,
		precision: str = "bf16",
		output_dir: str = "./outputs",
		gradient_checkpointing: bool = False,
		callbacks: list = [],
		metric_for_best_model: str = "eval_acc",
		warmup_ratio: float = 0.03,
		label_names: str = "labels",
		load_best_model_at_end: bool = True,
		overwrite_output_dir: bool = True,
	):
		self.model = model
		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.lr_scheduler_type = lr_scheduler_type
		self.scheduler = self.lr_scheduler_type if self.lr_scheduler_type is not None else None
		self.warmup_ratio = warmup_ratio
		self.max_grad_norm = max_grad_norm
		self.num_train_epochs = num_train_epochs
		self.eval_steps = eval_steps
		self.precision = precision
		self.output_dir = output_dir
		self.callbacks = callbacks or []
		self.label_names = label_names
		self.load_best_model_at_end = load_best_model_at_end
		self.metric_for_best_model = metric_for_best_model
		self.best_metric = -float("inf") if self.metric_for_best_model == "eval_acc" else float("inf")
		self.total_training_steps = None
		self.early_stopper = next((cb for cb in self.callbacks if isinstance(cb, EarlyStoppingCallback)), None)
		self.state = TrainerState()
		self.control = TrainerControl()
		self.epoch = 0
		self.state.epoch = 0
		self.global_step = 0
		self.args = CustomArgs(metric_for_best_model=self.metric_for_best_model, greater_is_better=(self.metric_for_best_model == "eval_acc"), load_best_model_at_end=self.load_best_model_at_end)
		if gradient_checkpointing:
			self.model.gradient_checkpointing_enable()
		if precision not in ["bf16", "fp16", "fp32"]:
			raise ValueError("precision must be one of 'bf16', 'fp16', or 'fp32'")
		self.scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
		self._init_output_dir(overwrite_output_dir)
		self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "logs", f"run-{self.state.global_step}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

	def _init_output_dir(self, overwrite_output_dir):
		if os.path.exists(self.output_dir):
			if overwrite_output_dir:
				print(f"Overwriting existing output directory: {self.output_dir}")
				shutil.rmtree(self.output_dir)
				os.makedirs(self.output_dir, exist_ok=True)
			else:
				latest_checkpoint = self._get_latest_checkpoint(self.output_dir)
				if latest_checkpoint:
					print(f"Resuming from checkpoint: {latest_checkpoint}")
					self.load_checkpoint(latest_checkpoint)
				else:
					print("No checkpoint found. Using provided configuration.")
		else:
			os.makedirs(self.output_dir, exist_ok=True)

	def _get_latest_checkpoint(self, base_dir):
		if not os.path.exists(base_dir):
			return None
		checkpoint_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
		if not checkpoint_dirs:
			return None
		checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
		return checkpoint_dirs[-1]

	def train(self, train_ds, eval_ds):
		if self.total_training_steps is None:
			self.total_training_steps = self.num_train_epochs * len(train_ds)

		if self.scheduler is not None:
			self.scheduler = get_scheduler(self.scheduler, optimizer=self.optimizer,
				num_warmup_steps=int(self.warmup_ratio * self.total_training_steps), num_training_steps=self.total_training_steps,)

		self.global_step = getattr(self, "global_step", getattr(self.state, "global_step", 0))
		self.epoch = getattr(self.state, "epoch", 0)
		progress_bar = tqdm(total=self.total_training_steps, desc="Training", initial=self.global_step)

		for epoch in range(self.epoch, self.num_train_epochs):
			self.model.train()
			_loss, _acc = 0.0, 0.0
			for batch in train_ds:
				self.global_step += 1
				self.state.global_step = self.global_step
				self.state.epoch = epoch

				metrics = self.train_batch(batch)
				_loss += metrics['loss']
				_acc += metrics['acc']

				self.writer.add_scalar("train/loss", metrics['loss'], self.global_step)
				self.writer.add_scalar("train/acc", metrics['acc'], self.global_step)
				progress_bar.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{100. * metrics['acc']:.2f}")
				progress_bar.update(1)

				if (self.global_step % self.eval_steps == 0) or (self.global_step == self.total_training_steps):
					eval_metrics = self.evaluate(eval_ds)
					print(f"\n[Step {self.global_step}/{self.total_training_steps}] - "
						f"train_loss: {_loss / self.global_step:.4f}, train_acc: {100. * _acc / self.global_step:.2f}% | "
						f"val_loss: {eval_metrics['eval_loss']:.4f}, val_acc: {100. * eval_metrics['eval_acc']:.2f}% | "
						f"lr: {self.optimizer.param_groups[0]['lr']:.6e}")

					self.writer.add_scalar("eval/loss", eval_metrics['eval_loss'], self.global_step)
					self.writer.add_scalar("eval/acc", eval_metrics['eval_acc'], self.global_step)

					current = eval_metrics[self.metric_for_best_model]
					improved = current > self.best_metric if self.metric_for_best_model == "eval_acc" else current < self.best_metric
					if improved:
						self.best_metric = current
						self.state.best_metric = current
						self.save_checkpoint(self.global_step)

					if self.early_stopper:
						self.early_stopper.on_evaluate(self.args, self.state, self.control, eval_metrics)
						if self.control.should_training_stop:
							print("Early stopping triggered.")
							progress_bar.close()
							return
		progress_bar.close()

	def train_batch(self, batch):
		self.optimizer.zero_grad()
		batch = {k: v.to(self.model.device, non_blocking=True) for k, v in batch.items()}
		autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16 if self.precision == "fp16" else torch.float32
		with torch.autocast("cuda", dtype=autocast_dtype):
			logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
			logits, labels = logits.view(-1, self.tokenizer._vocab_size), batch[self.label_names].view(-1)
			log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
			loss = torch.nn.functional.nll_loss(log_probs, labels, ignore_index=-100)

		if self.precision == "fp16":
			self.scaler.scale(loss).backward()
			self.scaler.unscale_(self.optimizer)
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
			self.scaler.step(self.optimizer)
			self.scaler.update()
		else:
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
			self.optimizer.step()
		if self.scheduler:
			self.scheduler.step()
		preds = torch.argmax(log_probs.detach(), dim=-1)
		acc = compute_cls_accuracy(preds, labels, self.tokenizer)
		return {"loss": loss.item(), "acc": acc}

	@torch.no_grad()
	def evaluate(self, ds):
		self.model.eval()
		_loss, _acc = 0.0, 0.0
		for batch in tqdm(ds, desc="Evaluating"):
			batch = {k: v.to(self.model.device, non_blocking=True) for k, v in batch.items()}
			logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
			logits, labels = logits.view(-1, self.tokenizer._vocab_size), batch[self.label_names].view(-1)
			log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
			loss = torch.nn.functional.nll_loss(log_probs, labels, ignore_index=-100)
			_loss += loss.item()
			_acc += compute_cls_accuracy(torch.argmax(log_probs, dim=-1), labels, self.tokenizer)
		return {"eval_loss": _loss / len(ds), "eval_acc": _acc / len(ds)}

	def save_checkpoint(self, step):
		path = os.path.join(self.output_dir, f"checkpoint-{step}")
		os.makedirs(path, exist_ok=True)
		# Save model and tokenizer
		self.model.save_pretrained(path)
		self.tokenizer.save_pretrained(path)
		# Save optimizer and scheduler state
		torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
		if self.scheduler:
			torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
		# Save scaler
		if self.precision in ["fp16", "bf16"]:
			torch.save(self.scaler.state_dict(), os.path.join(path, "scaler.pt"))
		# Save training args
		torch.save(self.args.__dict__, os.path.join(path, "training_args.bin"))
		# Save training state
		self.state.global_step = self.global_step
		self.state.epoch = self.epoch
		self.state.best_metric = self.best_metric
		torch.save(self.state, os.path.join(path, "trainer_state.pt"))
		# Scheduler
		with open(os.path.join(path, "scheduler_config.json"), "w") as f:
			json.dump({"name": self.lr_scheduler_type, "num_training_steps": self.total_training_steps, "num_warmup_steps": int(self.warmup_ratio * self.total_training_steps),}, f, indent=2)
		# Metadata
		with open(os.path.join(path, "metadata.json"), "w") as f:
			json.dump({"step": step, "epoch": self.epoch, "best_metric": self.best_metric, "precision": self.precision,}, f, indent=2)
		print(f"Checkpoint saved at: {path}")

	def load_checkpoint(self, checkpoint_path):
		self.model = self.model.from_pretrained(checkpoint_path)
		self.tokenizer = self.tokenizer.from_pretrained(checkpoint_path)
		optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
		if os.path.exists(optimizer_path):
			self.optimizer.load_state_dict(torch.load(optimizer_path))
		# Load Scheduler
		scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
		config_path = os.path.join(checkpoint_path, "scheduler_config.json")
		if os.path.exists(scheduler_path):
			if os.path.exists(config_path):
				with open(config_path, "r") as f: _cfg = json.load(f)
				print(_cfg)
				self.scheduler = get_scheduler(name=_cfg["name"], optimizer=self.optimizer,
					num_warmup_steps=_cfg["num_warmup_steps"], num_training_steps=_cfg["num_training_steps"],)
			else:
				self.scheduler = get_scheduler("cosine", self.optimizer, num_warmup_steps=0, num_training_steps=1)
			self.scheduler.load_state_dict(torch.load(scheduler_path))
		# Load AMP scaler (for fp16 or bf16)
		scaler_path = os.path.join(checkpoint_path, "scaler.pt")
		if os.path.exists(scaler_path) and self.precision in ["fp16", "bf16"]:
			self.scaler.load_state_dict(torch.load(scaler_path))
		# Load training args
		args_path = os.path.join(checkpoint_path, "training_args.bin")
		if os.path.exists(args_path):
			saved_args = torch.load(args_path)
			self.args.__dict__.update(saved_args)
			self.metric_for_best_model = self.args.metric_for_best_model
			self.load_best_model_at_end = self.args.load_best_model_at_end
		# Load training state
		state_path = os.path.join(checkpoint_path, "trainer_state.pt")
		if os.path.exists(state_path):
			self.state = torch.load(state_path)
			self.global_step, self.epoch = getattr(self.state, "global_step", 0), getattr(self.state, "epoch", 0)
			self.best_metric = getattr(self.state, "best_metric", -float("inf") if self.metric_for_best_model == "eval_acc" else float("inf"))
		else:
			print("⚠️ trainer_state.pt not found. Will start fresh state.")
			self.global_step, self.epoch = 0, 0
			self.best_metric = -float("inf") if self.metric_for_best_model == "eval_acc" else float("inf")

		print(f"Checkpoint loaded. Resuming from step={self.global_step}, epoch={self.epoch}, best_metric={self.best_metric:.4f}")

def compute_cls_accuracy(preds, labels, tokenizer, ignore_index=-100):
	mask = labels != ignore_index
	if not mask.any(): return 0.0
	preds_text = tokenizer.convert_ids_to_tokens(preds[mask].tolist(), skip_special_tokens=True)
	labels_text = tokenizer.convert_ids_to_tokens(labels[mask].tolist(), skip_special_tokens=True)
	return np.mean(np.array(preds_text) == np.array(labels_text)) if labels_text else 0.0

# NOTE: To use for multi-class labels
def compute_f1_micro(preds, labels, tokenizer, all_class_names, ignore_index=-100):
	mask = labels != ignore_index
	if not mask.any():
		return 0.0
	preds_text = tokenizer.convert_ids_to_tokens(preds[mask].tolist(), skip_special_tokens=True)
	labels_text = tokenizer.convert_ids_to_tokens(labels[mask].tolist(), skip_special_tokens=True)
	pred_str = "".join(preds_text)
	label_str = "".join(labels_text)
	pred_labels = [l.strip() for l in pred_str.split(",") if l.strip()]
	true_labels = [l.strip() for l in label_str.split(",") if l.strip()]
	label_set = {name: idx for idx, name in enumerate(all_class_names)}
	y_true = np.zeros(len(all_class_names), dtype=int)
	y_pred = np.zeros(len(all_class_names), dtype=int)
	for l in true_labels:
		if l in label_set:
			y_true[label_set[l]] = 1
	for l in pred_labels:
		if l in label_set:
			y_pred[label_set[l]] = 1
	return f1_score(y_true, y_pred, average="micro")