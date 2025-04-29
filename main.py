import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from .utils import compute_accuracy, load_model, load_tokenizer
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, default_data_collator

DS_DIR = "./datasets"
DS_NAME = "audiomnist"
MODEL_NAME = "Llasa-1B"
CODEC_NAME = "xcodec2"
PATIENCE = 3
EXPERIMENT_ID = f"{DS_NAME}_{MODEL_NAME}_{CODEC_NAME}"

########################################################
print(f"Loading dataset {DS_NAME}...")
data = load_from_disk(f"{DS_DIR}/{DS_NAME}_tok")
data.set_format(type="torch")
########################################################

########################################################
print("Loading model & tokenizer...")
model, tokenizer = load_model(), load_tokenizer()
model.config.use_cache = False
model.gradient_checkpointing_enable()
########################################################

########################################################################################################################################################################
training_args = TrainingArguments(output_dir=EXPERIMENT_ID,
	fp16=False, bf16=True, gradient_checkpointing=True,
	# Training parameters
	per_device_train_batch_size=1, per_device_eval_batch_size=1, eval_accumulation_steps=2, gradient_accumulation_steps=4, num_train_epochs=1, max_steps=-1,
	learning_rate=1e-5, weight_decay=0.01, adam_beta2=0.95, warmup_ratio=0.03, lr_scheduler_type="cosine",
	# Evaluation
	eval_strategy="steps", eval_steps=3000, save_strategy="steps", save_steps=3000, save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="accuracy", greater_is_better=True,
	# Other
	ddp_find_unused_parameters=False, logging_strategy="steps", logging_steps=10, report_to="none", disable_tqdm=False, remove_unused_columns=False,)
########################################################################################################################################################################

########################################################################################################################################################################
trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, train_dataset=data['train'], eval_dataset=data['validation'],
	data_collator=default_data_collator, compute_metrics=compute_accuracy, callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
########################################################################################################################################################################

########################################################################################################################################################################
print("Train {MODEL_NAME} on {DS_NAME}...")
trainer.train()
########################################################################################################################################################################