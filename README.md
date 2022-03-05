# SpeechMix

Explore different way to mix speech model(wav2vec2, hubert) and nlp model(BART,T5,GPT) together.

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

## base

python train.py --speech_model_config wav2vec2 \
--nlp_model_config facebook/bart-base \
--SpeechMixEED \
--dataset librispeech_asr \
--field clean \
--train_split train.100 \
--test_split validation \
--batch 4 \
--grad_accum 20 \
--epoch 30 \
--worker 15 \
--share_layer_ratio 0 \
--down_scale 1 \
--lr 4e-5 \
--warmup_steps 500 \
--wandb

## downscale 2/4/8

python train.py --speech_model_config wav2vec2 \
--nlp_model_config facebook/bart-base \
--SpeechMixEED \
--dataset librispeech_asr \
--field clean \
--train_split train.100 \
--test_split validation \
--batch 4 \
--grad_accum 20 \
--epoch 30 \
--worker 15 \
--share_layer_ratio 0 \
--down_scale 2 \
--lr 4e-5 \
--warmup_steps 500 \
--wandb

python train.py --speech_model_config wav2vec2 \
--nlp_model_config facebook/bart-base \
--SpeechMixEED \
--dataset librispeech_asr \
--field clean \
--train_split train.100 \
--test_split validation \
--batch 4 \
--grad_accum 20 \
--epoch 30 \
--worker 15 \
--share_layer_ratio 0 \
--down_scale 4 \
--lr 4e-5 \
--warmup_steps 500 \
--wandb

python train.py --speech_model_config wav2vec2 \
--nlp_model_config facebook/bart-base \
--SpeechMixEED \
--dataset librispeech_asr \
--field clean \
--train_split train.100 \
--test_split validation \
--batch 4 \
--grad_accum 20 \
--epoch 30 \
--worker 15 \
--share_layer_ratio 0 \
--down_scale 8 \
--lr 4e-5 \
--warmup_steps 500 \
--wandb