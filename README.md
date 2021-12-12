# SpeechMix

Explore different way to mix speech model(wav2vec2, hubert) and nlp model(BART,T5,GPT) together.

## Introduction

For the same input:

```python
from datasets import load_dataset
import soundfile as sf


# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

transcript = ds['text'][0]
speech = ds["speech"][0]
```

### Speech encoder NLP decoder

```python
model = SpeechMixED("facebook/wav2vec2-base-960h", "facebook/bart-large")

transcript_tensor = model.tokenizer(transcript, return_tensors="pt").input_ids
speech_tensor = model.processor(speech, return_tensors="pt").input_values

model(speech_tensor, transcript_tensor)
```

### Speech encoder NLP decoder only fine-tune on cross attention/projection/decoder embedding

```python
model = SpeechMixED("facebook/wav2vec2-base-960h", "facebook/bart-large", ftl=True)

transcript_tensor = model.tokenizer(transcript, return_tensors="pt").input_ids
speech_tensor = model.processor(speech, return_tensors="pt").input_values

model(speech_tensor, transcript_tensor)
```

### Speech encoder NLP encoder decoder

```python
model = SpeechMixEED("facebook/wav2vec2-base-960h", "facebook/bart-large")

transcript_tensor = model.tokenizer(transcript, return_tensors="pt").input_ids
speech_tensor = model.processor(speech, return_tensors="pt").input_values

model(speech_tensor, transcript_tensor)
```

### Speech encoder NLP encoder decoder only fine-tune on layer norm and attention

```python
model = SpeechMixEED("facebook/wav2vec2-base-960h", "facebook/bart-large", lna=True)

transcript_tensor = model.tokenizer(transcript, return_tensors="pt").input_ids
speech_tensor = model.processor(speech, return_tensors="pt").input_values

model(speech_tensor, transcript_tensor)
```

### Speech encoder NLP encoder decoder only fine-tune on speech encoder

```python
model = SpeechMixEED("facebook/wav2vec2-base-960h", "facebook/bart-large", fne=True)

transcript_tensor = model.tokenizer(transcript, return_tensors="pt").input_ids
speech_tensor = model.processor(speech, return_tensors="pt").input_values

model(speech_tensor, transcript_tensor)
```

## Installation

### pip install

```bash
pip install speechmix
```

### Build from source

git clone and cd into this project.

```bash
pip install -e .
```

## Example
usage:   
`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixEED --lna --dataset librispeech_asr --field clean --train_split train.100 --test_split validation --batch 3 --grad_accum 8`   

`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixEED --fne --dataset librispeech_asr --field clean --train_split train.100 --test_split validation --batch 3 --grad_accum 8`    

`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixED --dataset librispeech_asr --field other --train_split train.500 --test_split validation --batch 3 --grad_accum 8`    

`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixED --ftl --dataset librispeech_asr --field other --train_split train.500 --test_split validation --batch 3 --grad_accum 8`   

`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixSelf  --dataset librispeech_asr --field clean --train_split train.100 --test_split validation --batch 3 --grad_accum 10`   

`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixGAN  --dataset librispeech_asr --field clean --train_split train.100 --test_split validation --batch 3 --grad_accum 10`   

`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixSelf  --dataset common_voice --field en --train_split train --test_split test --batch 5 --grad_accum 8`   

`python train.py --speech_model_config facebook/wav2vec2-large-robust-ft-libri-960h --nlp_model_config facebook/mbart-large-50-one-to-many-mmt --SpeechMixEED --lna --dataset patrickvonplaten/librispeech_asr_dummy --field clean --train_split validation --test_split test --batch 3 --grad_accum 4`

`python train.py --speech_model_config microsoft/unispeech-sat-large --nlp_model_config bigscience/T0_3B --SpeechMixSelf  --dataset librispeech_asr --field clean --train_split train.100 --test_split validation --batch 9 --grad_accum 3`

`python -m torch.distributed.launch --nproc_per_node=2 train.py --speech_model_config microsoft/unispeech-sat-large --nlp_model_config bigscience/T0_3B --SpeechMixSelf  --dataset librispeech_asr --field clean --train_split train.100 --test_split validation --batch 3 --grad_accum 6 --fne`