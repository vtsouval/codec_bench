import torch
import datasets
try:
	from .utils import BaseProcessorFn, preprocess_samples, load_tokenizer, load_codec, train_test_split
except ImportError:
	from utils import BaseProcessorFn, preprocess_samples, load_tokenizer, load_codec, train_test_split

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

def load_ds(hf_repo="vtsouval/flusense", preprocess_fn=preprocess_samples, label_key='label', **kwargs):
	_exclude_labels = {"etc", "vomit", "snore", "wheeze"}
	_exclude_indices = [7, 37, 53, 55, 57, 59, 61, 80, 82, 383, 419, 425, 435, 442, 448, 527, 530, 536, 609, 617, 655, 762, 1211, 1398, 1421, 1430, 1431, 1443, 1445, 1447, 1490, 1555, 1777, 1788, 1801, 1819, 1858, 1864, 1892, 1927, 1972, 1978, 2032, 2202, 2256, 2554, 2627, 2641, 2693, 2739, 2743, 2745, 2754, 2755, 2853, 2883, 3017, 3083, 3085, 3087, 3104, 3120, 3131, 3195, 3229, 3231, 3286, 3304, 3323, 3372, 3485, 3490, 3509, 3654, 3688, 3702, 3704, 3706, 3713, 3714, 3971, 4028, 4080, 4154, 4158, 4163, 4193, 4248, 4286, 4362, 4364, 4391, 4403, 4433, 4439, 4445, 4449, 4724, 4736, 4833, 4946, 5097, 5440, 5557, 5578, 5633, 5692, 5993, 6053, 6056, 6059, 6239, 6244, 6245, 6246, 6249, 6258, 6402, 6577, 6616, 6648, 6660, 6691, 6876, 6913, 6924, 7028, 7079, 7291, 7784, 8169, 8209, 8215, 8217, 8232, 8374, 8379, 8406, 8410, 8414, 8416, 8417, 8419, 8538, 8560, 8566, 8599, 8604, 9115, 9470, 9478, 9556, 9642, 9646, 9656, 9708, 10051, 10052, 10116, 10447, 10449, 10655, 10657, 10661, 11131, 11660]
	ds = datasets.load_dataset(hf_repo, trust_remote_code=True)
	ds['train'] = ds['train'].select([i for i in range(len(ds['train'])) if i not in _exclude_indices])
	ds = ds.filter(lambda x: x[label_key] not in _exclude_labels)
	ds = train_test_split(ds, label_key=label_key, test_size=0.1, format_to_torch=False)
	ds['train'] = ds['train'].remove_columns([c for c in ds['train'].column_names if c not in ['audio', label_key]])
	ds['val'] = ds['val'].remove_columns([c for c in ds['val'].column_names if c not in ['audio', label_key]])
	ds['test'] = ds['test'].remove_columns([c for c in ds['test'].column_names if c not in ['audio', label_key]])
	class_names = list(set(ds["train"][label_key]))
	ds = ds.map(lambda x: preprocess_fn(x, label_key=label_key, one_hot_labels=False), remove_columns=ds["train"].column_names).with_format("torch")
	return ds, class_names

if __name__ == "__main__":
	ds, class_names = load_ds(hf_repo="vtsouval/flusense")
	audio_codec, tokenizer = load_codec(), load_tokenizer()
	preprocess_fn = ProcessorFn(class_names, audio_codec, tokenizer)
	ds = ds.map(preprocess_fn, remove_columns=["waveform", "label"], batched=False).with_format("torch")
	ds.save_to_disk("../flusense")

