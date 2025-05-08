import torch
import datasets
try:
	from .utils import BaseProcessorFn, preprocess_samples, load_tokenizer, load_codec
except ImportError:
	from utils import BaseProcessorFn, preprocess_samples, load_tokenizer, load_codec

class ProcessorFn(BaseProcessorFn):

	def __init__(self, class_names, audio_codec, tokenizer, max_length=2048, prompt="Classify the audio in the following segment.", device="cuda", **kwargs):
		super().__init__(max_length=max_length, **kwargs)
		self._convert_classes_to_tokens(tokenizer, class_names, prompt)
		self.device = device
		self.pad_token_id = tokenizer.pad_token_id
		self.audio_codec = audio_codec.to(self.device)
		self.sampling_rate = audio_codec.sampling_rate

	def _convert_classes_to_tokens(self, tokenizer, class_names, prompt):
		# Define text tokens for each class
		self.label_to_tokens = {
			label: tokenizer.apply_chat_template(
				[
					{"role": "user", "content": f"<|TEXT_UNDERSTANDING_START|>{prompt}<|TEXT_UNDERSTANDING_END|>"},
					{"role": "assistant", "content": label}
				],
				tokenize=True,
				continue_final_message=False
			)
			for label in class_names
		}
		# Define text tokens len (without label)
		self.extra_pad = len(
			tokenizer.apply_chat_template(
				[
					{"role": "user", "content": f"<|TEXT_UNDERSTANDING_START|>{prompt}<|TEXT_UNDERSTANDING_END|>"}
				],
			tokenize=True, add_generation_prompt=True))

	def extract_audio_tokens(self, x, base_num=128264, start_token_id=128262, end_token_id=128263):
		if x.ndim == 1:
			x = x.unsqueeze(0)
		with torch.no_grad():
			_padding = (320 - (x.shape[1] % 320)) % 320
			x = torch.nn.functional.pad(x, (0, _padding))
			wav2vec_in = torch.nn.functional.pad(x[0, :], (160, 160))
			wav2vec_feat = self.audio_codec.feature_extractor(wav2vec_in, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
			wav2vec_feat = self.audio_codec.semantic_model(wav2vec_feat).hidden_states[16].transpose(1, 2)
			wav2vec_feat = self.audio_codec.SemanticEncoder_module(wav2vec_feat)
			x = x.to(self.device)
			codec_feat = self.audio_codec.CodecEnc(x.unsqueeze(1)).transpose(1, 2)
			speech_tokens = self.audio_codec.fc_prior(torch.cat([wav2vec_feat, codec_feat], dim=1).transpose(1, 2)).transpose(1, 2)
			speech_tokens = self.audio_codec.generator(speech_tokens, vq=True)[1]
		if speech_tokens.dim() == 3 and speech_tokens.size(1) == 1:
			speech_tokens = speech_tokens.squeeze(1)
		return [start_token_id] + list((speech_tokens.cpu() + base_num).view(-1)) + [end_token_id]

	def extract_text_tokens(self, x):
		return self.label_to_tokens[x]

	def __call__(self, x):
		audio_tk = self.extract_audio_tokens(x["waveform"])
		text_tk = self.extract_text_tokens(x['label'])
		return self._combine_tokens(x_0=audio_tk, x_1=text_tk, pad_token_id=self.pad_token_id, extra_pad=self.extra_pad, pad_to_max_length=False)

def load_ds(hf_repo="danavery/urbansound8K", fold=1, preprocess_fn=preprocess_samples, label_key='class'):
	_map = {
		1: {'train': [3, 4, 5, 6, 7, 8, 9, 10], 'val': [2], 'test': [1]},
		2: {'train': [1, 3, 4, 5, 6, 7, 8, 9], 'val': [3], 'test': [2]},
		3: {'train': [1, 2, 4, 5, 6, 7, 8, 9], 'val': [4], 'test': [3]},
		4: {'train': [1, 2, 3, 5, 6, 7, 8, 9], 'val': [5], 'test': [4]},
		5: {'train': [1, 2, 3, 4, 6, 7, 8, 9], 'val': [6], 'test': [5]},
		6: {'train': [1, 2, 3, 4, 5, 7, 8, 9], 'val': [7], 'test': [6]},
		7: {'train': [1, 2, 3, 4, 5, 6, 8, 9], 'val': [8], 'test': [7]},
		8: {'train': [1, 2, 3, 4, 5, 6, 7, 9], 'val': [9], 'test': [8]},
		9: {'train': [1, 2, 3, 4, 5, 6, 7, 8], 'val': [10], 'test': [9]},
		10: {'train': [1, 2, 3, 4, 5, 6, 7, 8], 'val': [1], 'test': [10]},
	}
	ds = datasets.load_dataset(hf_repo)['train'].map(lambda e: {label_key: e[label_key].replace("_", " ")})
	ds = datasets.DatasetDict({'train': ds.filter(lambda e: e['fold'] in  _map[fold]['train']), 'val': ds.filter(lambda e: e['fold'] in  _map[fold]['val']), 'test': ds.filter(lambda e: e['fold'] in  _map[fold]['test'])})
	ds['train'] = ds['train'].remove_columns([c for c in ds['train'].column_names if c not in ['audio', label_key]])
	ds['val'] = ds['val'].remove_columns([c for c in ds['val'].column_names if c not in ['audio', label_key]])
	ds['test'] = ds['test'].remove_columns([c for c in ds['test'].column_names if c not in ['audio', label_key]])
	class_names = list(set(ds["train"][label_key]))
	ds = ds.map(lambda x: preprocess_fn(x, label_key=label_key, one_hot_labels=False), remove_columns=ds["train"].column_names).with_format("torch")
	return ds, class_names

if __name__ == "__main__":
	ds, class_names = load_ds(hf_repo="danavery/urbansound8K")
	audio_codec, tokenizer = load_codec(), load_tokenizer()
	preprocess_fn = ProcessorFn(class_names, audio_codec, tokenizer)
	ds = ds.map(preprocess_fn, remove_columns=["waveform", "label"], batched=False).with_format("torch")
	ds.save_to_disk("../us8k")

