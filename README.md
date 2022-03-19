# SpeechMix

Explore different way to mix speech model(wav2vec2, hubert) and nlp model(BART,T5,GPT) together.   

Implementation of:   
[Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) - ACL2021   
[Multilingual Speech Translation with Efficient Finetuning of Pretrained Models](https://arxiv.org/abs/2010.12829) - ACL2021   
[Lightweight Adapter Tuning for Multilingual Speech Translation](https://arxiv.org/abs/2106.01463) - Interspeech 2021   
[Improving Speech Translation by Understanding and Learning from the Auxiliary Text Translation Task](https://arxiv.org/abs/2107.05782) - ACL2021   
[A General Multi-Task Learning Framework to Leverage Text Data for Speech to Text Tasks](https://arxiv.org/abs/2010.11338) - ICASSP 2021   

## Installation

### pip install

```bash
pip install speechmix
```

### Build from source

git clone and cd into this project.

```shell
pip install -e .
```

## Name the project(!important)
WANDB_PROJECT=amazing

## base
```shell
python train.py --speech_model_config wav2vec2 \
--nlp_model_config facebook/bart-base \
--SpeechMixEED \
--dataset librispeech_asr \
--field clean \
--train_split train.100 \
--test_split validation \
--batch 3 \
--grad_accum 20 \
--epoch 30 \
--worker 15 \
--share_layer_ratio 0 \
--down_scale 2 \
--lr 4e-5 \
--warmup_steps 500 \
--wandb \
--notes base
```


