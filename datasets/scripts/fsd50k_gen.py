import torch
import datasets
import itertools
from sentence_transformers import SentenceTransformer, util
try:
	from .utils import BaseProcessorFn, preprocess_samples, load_tokenizer, load_codec
except ImportError:
	from utils import BaseProcessorFn, preprocess_samples, load_tokenizer, load_codec

MAX_SAMPLES = int(30 * 160000)

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
		x["waveform"] = x["waveform"][:MAX_SAMPLES]
		audio_tk = self.extract_audio_tokens(x["waveform"])
		text_tk = self.extract_text_tokens(x['label'])
		return self._combine_tokens(x_0=audio_tk, x_1=text_tk, pad_token_id=self.pad_token_id, extra_pad=self.extra_pad, pad_to_max_length=False)

def _compute_labels(example, available_classes, class_embeddings, threshold=0.6, top_k=4):
	model = SentenceTransformer("all-MiniLM-L6-v2")
	label_embedding = model.encode(example["text"].replace("The sounds of ", "").lower(), convert_to_tensor=True)
	similarities = util.cos_sim(label_embedding, class_embeddings)[0]
	top_indices = (similarities >= threshold).nonzero(as_tuple=True)[0]
	if len(top_indices) == 0:
		top_indices = [int(similarities.argmax())]
	elif len(top_indices) > top_k:
		top_indices = similarities[top_indices].topk(top_k).indices
	matched_labels = [available_classes[int(i)] for i in top_indices]
	return {"label": ", ".join(matched_labels)}

def _get_class_embeddigns():
	model = SentenceTransformer("all-MiniLM-L6-v2")
	all_classes = ["Accelerating_and_revving_and_vroom","Accordion","Acoustic_guitar","Aircraft","Alarm","Animal","Applause","Bark","Bass_drum","Bass_guitar","Bathtub_(filling_or_washing)","Bell","Bicycle","Bicycle_bell","Bird","Bird_vocalization_and_bird_call_and_bird_song","Boat_and_Water_vehicle","Boiling","Boom","Bowed_string_instrument","Brass_instrument","Breathing","Burping_and_eructation","Bus","Buzz","Camera","Car","Car_passing_by","Cat","Chatter","Cheering","Chewing_and_mastication","Chicken_and_rooster","Child_speech_and_kid_speaking","Chime","Chink_and_clink","Chirp_and_tweet","Chuckle_and_chortle","Church_bell","Clapping","Clock","Coin_(dropping)","Computer_keyboard","Conversation","Cough","Cowbell","Crack","Crackle","Crash_cymbal","Cricket","Crow","Crowd","Crumpling_and_crinkling","Crushing","Crying_and_sobbing","Cupboard_open_or_close","Cutlery_and_silverware","Cymbal","Dishes_and_pots_and_pans","Dog","Domestic_animals_and_pets","Domestic_sounds_and_home_sounds","Door","Doorbell","Drawer_open_or_close","Drill","Drip","Drum","Drum_kit","Electric_guitar","Engine","Engine_starting","Explosion","Fart","Female_singing","Female_speech_and_woman_speaking","Fill_(with_liquid)","Finger_snapping","Fire","Fireworks","Fixed-wing_aircraft_and_airplane","Fowl","Frog","Frying_(food)","Gasp","Giggle","Glass","Glockenspiel","Gong","Growling","Guitar","Gull_and_seagull","Gunshot_and_gunfire","Gurgling","Hammer","Hands","Harmonica","Harp","Hi-hat","Hiss","Human_group_actions","Human_voice","Idling","Insect","Keyboard_(musical)","Keys_jangling","Knock","Laughter","Liquid","Livestock_and_farm_animals_and_working_animals","Male_singing","Male_speech_and_man_speaking","Mallet_percussion","Marimba_and_xylophone","Mechanical_fan","Mechanisms","Meow","Microwave_oven","Motor_vehicle_(road)","Motorcycle","Music","Musical_instrument","Ocean","Organ","Packing_tape_and_duct_tape","Percussion","Piano","Plucked_string_instrument","Pour","Power_tool","Printer","Purr","Race_car_and_auto_racing","Rail_transport","Rain","Raindrop","Ratchet_and_pawl","Rattle","Rattle_(instrument)","Respiratory_sounds","Ringtone","Run","Sawing","Scissors","Scratching_(performance_technique)","Screaming","Screech","Shatter","Shout","Sigh","Singing","Sink_(filling_or_washing)","Siren","Skateboard","Slam","Sliding_door","Snare_drum","Sneeze","Speech","Speech_synthesizer","Splash_and_splatter","Squeak","Stream","Strum","Subway_and_metro_and_underground","Tabla","Tambourine","Tap","Tearing","Telephone","Thump_and_thud","Thunder","Thunderstorm","Tick","Tick-tock","Toilet_flush","Tools","Traffic_noise_and_roadway_noise","Train","Trickle_and_dribble","Truck","Trumpet","Typewriter","Typing","Vehicle","Vehicle_horn_and_car_horn_and_honking","Walk_and_footsteps","Water","Water_tap_and_faucet","Waves_and_surf","Whispering","Whoosh_and_swoosh_and_swish","Wild_animals","Wind","Wind_chime","Wind_instrument_and_woodwind_instrument","Wood","Writing","Yell","Zipper_(clothing)"]
	all_classes = sorted(set([label.replace("_", " ").lower() for label in all_labels]))
	return all_classes, model.encode(all_classes, convert_to_tensor=True)

def load_ds(hf_repo="CLAPv2/FSD50K", preprocess_fn=preprocess_samples, label_key='label'):
	ds = datasets.load_dataset(hf_repo)
	ds = datasets.DatasetDict({"train": ds["train"], "val": ds["validation"], "test": ds["test"]})
	class_names, class_embeddings = _get_class_embeddigns()
	ds = ds.map(_compute_labels, fn_kwargs={"available_classes": class_names, "class_embeddings": class_embeddings, "threshold": 0.7, "top_k": 5})
	all_labels = sorted(set(l for split in ['train', 'val', 'test'] for l in ds[split]["label"]))
	class_names = list(set(list(itertools.chain.from_iterable([x.strip() for x in l.split(",")] for split in ['train', 'val', 'test'] for l in ds[split]["label"]))))
	ds = ds.map(lambda x: preprocess_fn(x, label_key=label_key, one_hot_labels=False), remove_columns=ds["train"].column_names).with_format("torch")
	return ds, all_labels, class_names

if __name__ == "__main__":
	ds, all_labels, class_names = load_ds(hf_repo="CLAPv2/FSD50K")
	audio_codec, tokenizer = load_codec(), load_tokenizer()
	preprocess_fn = ProcessorFn(all_labels, audio_codec, tokenizer)
	ds = ds.map(preprocess_fn, remove_columns=["waveform", "label"], batched=False).with_format("torch")
	ds.save_to_disk("../fsd50k")