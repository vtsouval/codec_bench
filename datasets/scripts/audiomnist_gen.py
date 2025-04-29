import torch
import torchaudio
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model

def load_audio_codec(model="HKUSTAudio/xcodec2"):
	codec = XCodec2Model.from_pretrained(model)
	codec.sampling_rate = 16000
	codec.max_audio_duration = 41.0
	codec.max_audio_len = int(codec.sampling_rate * codec.max_audio_duration)
	codec.eval()
	for _,p in codec.named_parameters(): p.requires_grad = False
	return codec

def load_tokenizer(model="HKUSTAudio/Llasa-1B-Multilingual"):
	tokenizer = AutoTokenizer.from_pretrained(model, padding_side="right", use_fast=True, local_files_only=True)
	tokenizer.text_vocab_size = 128000
	tokenizer.speech_vocab_size = 65536
	tokenizer.reserved_tokens = 263
	return tokenizer

class ProcessorFn:

	def __init__(self, audio_codec, tokenizer, max_length=2048, prompt="Classify the audio in the following segment.", device="cuda:1"):
		self.device = device
		self.audio_codec = audio_codec.to(self.device)
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.prompt = prompt
		self.sampling_rate = audio_codec.sampling_rate
		self.prompt_len = len(self.tokenizer.apply_chat_template([{"role": "user", "content": f"<|TEXT_UNDERSTANDING_START|>{self.prompt}<|TEXT_UNDERSTANDING_END|>"},{"role": "assistant", "content":""}], tokenize=True, add_generation_prompt=False))

	def extract_audio_tokens(self, x, base_num=128264, start_token_id=128262, end_token_id=128263):
		if x.ndim == 1:
			x = x.unsqueeze(0)
		with torch.no_grad():
			_padding = (320 - (x.shape[1] % 320))
			x = torch.nn.functional.pad(x, (0, _padding))

			wav2vec_in = torch.nn.functional.pad(x[0,:], (160, 160))
			wav2vec_feat = self.audio_codec.feature_extractor(wav2vec_in, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
			wav2vec_feat = self.audio_codec.semantic_model(wav2vec_feat).hidden_states[16].transpose(1, 2)
			wav2vec_feat = self.audio_codec.SemanticEncoder_module(wav2vec_feat)

			x = x.to(self.device)
			codec_feat = self.audio_codec.CodecEnc(x.unsqueeze(1)).transpose(1, 2)
			speech_tokens = self.audio_codec.fc_prior(torch.cat([wav2vec_feat, codec_feat], dim=1).transpose(1, 2)).transpose(1, 2)
			speech_tokens = self.audio_codec.generator(speech_tokens, vq=True)[1]

		if speech_tokens.dim() == 3 and speech_tokens.size(1) == 1:
			speech_tokens = speech_tokens.squeeze(1)

		speech_tokens = [start_token_id] + list((speech_tokens.cpu() + base_num).view(-1)) + [end_token_id]
		return speech_tokens

	def extract_text_tokens_from_label(self, label):
		chat = [{"role": "user", "content": f"<|TEXT_UNDERSTANDING_START|>{self.prompt}<|TEXT_UNDERSTANDING_END|>"},{"role":"assistant", "content":label}]
		return self.tokenizer.apply_chat_template(chat, tokenize=True, continue_final_message=True)

	def combine_text_audio_tokens(self, text_tokens, speech_tokens, ignore_index=-100, extra_pad=0):
		input_ids = speech_tokens + text_tokens
		input_ids = input_ids[:self.max_length] if len(input_ids) > self.max_length else input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
		input_ids = torch.tensor(input_ids, dtype=torch.long)
		attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
		labels = input_ids.clone()
		labels[:len(speech_tokens) + self.prompt_len + extra_pad] = ignore_index
		labels[input_ids == self.tokenizer.pad_token_id] = ignore_index
		return input_ids, attention_mask, labels

	def __call__(self, x):
		speech_tokens = self.extract_audio_tokens(x["waveform"])
		text_tokens = self.extract_text_tokens_from_label(str(x["label"].item()))
		input_ids, attention_mask, labels = self.combine_text_audio_tokens(text_tokens, speech_tokens)
		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,}

def _preprocess_fn(x, target_sr=16000):
	waveform = x["audio"]["array"].clone().detach().float().unsqueeze(0)
	sr = x["audio"]["sampling_rate"]
	if sr != target_sr:
		waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
	return {"waveform": waveform.squeeze(0), "label": int(x["digit"])}

def load_audiomnist(hf_repo="gilkeyio/AudioMNIST", preprocess_fn=_preprocess_fn, max_size=2000):
	dataset = load_dataset(hf_repo)
	dataset.set_format(type="torch")
	dataset = dataset.map(preprocess_fn, remove_columns=dataset["train"].column_names)
	labels = dataset["train"]["label"]
	class_names = set(labels)
	train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=0.1, stratify=labels, random_state=42)
	train_dataset = dataset["train"].select(train_idx)
	val_dataset = dataset["train"].select(val_idx)
	test_dataset = dataset["test"]
	if max_size is not None:
		train_dataset = train_dataset.select(range(min(max_size, len(train_dataset))))
		val_dataset = val_dataset.select(range(min(max_size//4, len(val_dataset))))
		test_dataset = test_dataset.select(range(min(max_size//4, len(test_dataset))))
	dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})
	dataset_dict.set_format(type="torch")
	return dataset_dict, class_names

print("Load dataset...")
data, _ = load_audiomnist(max_size=None)
print("Load audio codec...")
audio_codec = load_audio_codec()
print("Load text tokenizer...")
tokenizer = load_tokenizer()
print("Create tokenization fn...")
tokenizer_fn = ProcessorFn(audio_codec=audio_codec, tokenizer=tokenizer)
print("Tokenize datadset...")
data = data.map(tokenizer_fn, remove_columns=["waveform", "label"], batched=False)
data.save_to_disk("../audiomnist_tok")