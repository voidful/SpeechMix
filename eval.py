import torch
from datasets import load_dataset

from speechmix import HFSpeechMixEED

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
spm = HFSpeechMixEED('facebook/wav2vec2-base-960h', "voidful/bart-base-chinese",
                     fixed_parameters=False, share_layer_ratio=0.5, down_scale=8,
                     weighted_sum=False)
outputs = spm.generate(torch.tensor([ds[0]["audio"]["array"]], device=spm.device))
decoded = spm.tokenizer.decode(outputs[0], skip_special_tokens=True)
print("decoded", decoded)
