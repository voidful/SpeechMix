import torch
from datasets import load_dataset

from speechmix import HFSpeechMixEED

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
spm = HFSpeechMixEED('facebook/wav2vec2-large-960h-lv60', "voidful/phoneme_byt5",
                     fixed_parameters=False, share_layer_ratio=0, down_scale=8,
                     weighted_sum=False)
spm.load_state_dict(torch.load('./pytorch_model.bin'))
spm.eval()
outputs = spm.generate(torch.tensor([ds[0]["audio"]["array"]], device=spm.device),
                       decoder_text_prompt='tranlate en -> de:', max_length=250)
decoded = spm.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(ds[0]['text'])
print("decoded", decoded)
