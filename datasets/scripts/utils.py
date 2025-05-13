
import torch
import torchaudio
import datasets
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model
from sklearn.model_selection import train_test_split as _train_test_split
from scipy.signal import resample

DATASET_PROMPT = {
    # Speech
	'audiomnist': "Identify the spoken digit in the following audio recording.",
	'cremad': "Classify the emotion expressed in the following speech audio.",
	'emovo': "Classify the emotion expressed in the following speech audio.",
	'ravdess': "Classify the emotion expressed in the following speech audio.",
	'iemocap': "Classify the emotion expressed in the following speech audio.",
	'slurp': "Determine the intent behind the spoken command in the following audio segment.",
	'timit-dialect': "Identify the speaker's dialect region based on the following audio sample.",
	'vctk': "Identify the speaker's identity based on the following audio sample.",
	# Paralinguistic / Environmental
	'esc50': "Classify the environmental sound in the following audio recording.",
	'us8k': "Classify the urban sound present in the following audio recording.",
	'arca23k-fsd': "Identify the sound event present in the following audio segment.",
	'fsd50k': "Identify the sound event present in the following audio segment.",
	'vivae': "Classify the vocal emotion expressed in the following audio recording.",
	'flusense': "Identify illness-related sounds such as cough or sneeze in the following audio segment.",
	# MUSIC
	'medley-solos-db': "Identify the musical instrument being played in the following audio segment.",
	'fma': "Classify the musical genre of the following audio segment.",
	'gtzan': "Classify the musical genre of the following audio segment.",
	'mgt_genre': "Classify the musical genre of the following audio segment.",
	'mgt_mood': "Classify the mood expressed in the following music segment.",
}

def load_codec(model="HKUSTAudio/xcodec2"):
	codec = XCodec2Model.from_pretrained(model)
	codec.sampling_rate = 16000
	codec.max_audio_duration = 41.0
	codec.max_audio_len = int(codec.sampling_rate * codec.max_audio_duration)
	codec.eval()
	for _,p in codec.named_parameters(): p.requires_grad = False
	return codec

def load_tokenizer(model="HKUSTAudio/Llasa-1B-Multilingual"):
	tokenizer = AutoTokenizer.from_pretrained(model, padding_side="right", use_fast=True, local_files_only=True)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.text_vocab_size = 128000
	tokenizer.speech_vocab_size = 65536
	tokenizer.reserved_tokens = 263
	return tokenizer

def preprocess_samples_torch(x, target_sr=16000, label_key="digit"):
	waveform = x["audio"]["array"].float()
	sr = x["audio"]["sampling_rate"]
	if waveform.dim() == 1:
		waveform = waveform.unsqueeze(0)
	if waveform.size(0) > 1:
		waveform = waveform.mean(dim=0, keepdim=True)
	if sr != target_sr:
		waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
	return {"waveform": waveform.squeeze(0), "label": int(x[label_key]),}

def preprocess_samples(x, target_sr=16000, label_key="digit", one_hot_labels=True):
	waveform = x["audio"]["array"]
	sr = x["audio"]["sampling_rate"]
	if waveform.ndim == 1:
		waveform = waveform[np.newaxis, :]
	if waveform.shape[0] > 1:
		waveform = waveform.mean(axis=0, keepdims=True)
	if sr != target_sr:
		num_samples = int(waveform.shape[1] * target_sr / sr)
		waveform = resample(waveform, num=num_samples, axis=1)
	return {"waveform": waveform.squeeze(), "label": int(x[label_key]) if one_hot_labels else x[label_key]}

def train_test_split(ds, label_key='label', max_size=None, test_size=0.05, val_size=0.05, format_to_torch=True, seed=42):
	labels = ds["train"][label_key]
	train_idx, temp_idx = _train_test_split(list(range(len(labels))), test_size=test_size + val_size if "test" not in ds else val_size, stratify=labels, random_state=seed,)
	train_ds = ds["train"].select(train_idx)
	if "test" in ds:
		test_ds = ds["test"]
		val_ds = ds["train"].select(temp_idx)
	else:
		temp_labels = [labels[i] for i in temp_idx]
		val_idx, test_idx = _train_test_split(temp_idx, test_size=test_size / (test_size + val_size), stratify=temp_labels, random_state=seed,)
		val_ds = ds["train"].select(val_idx)
		test_ds = ds["train"].select(test_idx)
	if max_size is not None:
		train_ds = train_ds.select(range(min(max_size, len(train_ds))))
		val_ds = val_ds.select(range(min(max_size // 2, len(val_ds))))
		test_ds = test_ds.select(range(min(max_size // 2, len(test_ds))))
	if format_to_torch:
		return datasets.DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds,}).with_format("torch")
	else:
		return datasets.DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds,})

class BaseProcessorFn(ABC):

	def __init__(self, max_length=2048, ignore_index=-100):
		self.max_length = max_length
		self.ignore_index = ignore_index
		self.label_to_tokens = None
		self.extra_pad = 0

	@abstractmethod
	def extract_audio_tokens(self, x):
		pass

	@abstractmethod
	def extract_text_tokens(self, x):
		pass

	@abstractmethod
	def __call__(self, x):
		pass

	def _pad_or_crop(self, x, pad_token_id):
		if len(x) > self.max_length:
			return x[:self.max_length]
		else:
			return x + [pad_token_id] * (self.max_length - len(x))

	def _combine_tokens(self, x_0, x_1, pad_token_id, extra_pad, pad_to_max_length=False):
		if pad_to_max_length:
			input_ids = torch.tensor(self._pad_or_crop(x=x_0+x_1, pad_token_id=pad_token_id), dtype=torch.long)
		else:
			input_ids = torch.tensor(x_0+x_1, dtype=torch.long)
		attention_mask = (input_ids != pad_token_id).long()
		labels = input_ids.clone()
		labels[:len(x_0) + extra_pad] = self.ignore_index
		labels[input_ids == pad_token_id] = self.ignore_index
		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,}

