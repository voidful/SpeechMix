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
usage
`python train.py --speech_model_config voidful/wav2vec2-large-xlsr-53-tw-gpt --nlp_model_config voidful/bart-base-chinese --SpeechMixEED --fne`