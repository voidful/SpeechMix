import torch
from datasets import load_dataset

from speechmix import HFSpeechMixEED, SpeechMixEEDT5eval

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy",
                  "clean", split="validation")
spm = HFSpeechMixEED('facebook/wav2vec2-large-960h-lv60', "voidful/phoneme_byt5",
                     fixed_parameters=False, share_layer_ratio=0, down_scale=8,
                     weighted_sum=False)
spm.load_state_dict(torch.load('./pytorch_model.bin'))
spm.eval()


class STDataset(Dataset):
    def __init__(self, splt="train", tokenizer="google/mt5-small", translate_from="en", translate_to="de"):
        self.data = load_dataset(
            "patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        self.encoder_processor = AutoProcessor.from_pretrained(
            "facebook/wav2vec2-large-lv60", sampling_rate=16000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ii):
        idx = self.data[ii]
        speech_input = self.data[idx]["audio"]["array"]

        return {
            "speech_input": speech_input
        }


ds2 = STDataset()


def collate_batch(batch: List):
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    speech_input = [example['speech_input'] for example in batch]
    input_values = encoder_processor(
        speech_input, return_tensors="pt", padding="longest", sampling_rate=16000).input_values

    return {
        'speech_input': input_values
    }


valid_loader = DataLoader(
    ds2, batch_size=1, collate_fn=collate_batch, num_workers=32)
spm = spm.cuda()
for batch in tqdm(valid_loader):
    # for SpeechEEDT5eval
    outputs = spm(batch["speech_input"].to(model.device))

    # for HFSpeechEED
    outputs = spm.generate(batch["speech_input"].to(
        spm.device), max_length=250, num_beams=10)
#decoded = spm.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(ds[0]['text'])
print("decoded", decoded)
