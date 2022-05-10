import unittest
import pytest
from datasets import load_dataset
import torch
from speechmix import HFSpeechMixEED


class TestModel(unittest.TestCase):
    def test_hf(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        spm = HFSpeechMixEED('facebook/wav2vec2-base-960h', "voidful/bart-base-chinese",
                             fixed_parameters=False, share_layer_ratio=0.5, down_scale=4,
                             weighted_sum=False)
        outputs = spm.generate(torch.tensor([ds[0]["audio"]["array"]], device=spm.device))
        decoded = spm.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded)  # Machine learning is great, isn't it?

    def test_pure(self):
        spm = HFSpeechMixEED('facebook/wav2vec2-base-960h', "voidful/bart-base-chinese",
                             fixed_parameters=False, share_layer_ratio=0, down_scale=8,
                             weighted_sum=False)
        self.assertEqual(spm.speech_encoder_layer, 12)
        self.assertEqual(spm.nlp_encoder_layer, 6)
        self.assertEqual(len(spm.list_no_grad), 0)

    def test_share_layer(self):
        for share_layers_pair in [(1, 0), (0.5, 6), (0, 12)]:
            spm = HFSpeechMixEED('facebook/wav2vec2-base-960h', "voidful/bart-base-chinese",
                                 fixed_parameters=False, share_layer_ratio=share_layers_pair[0], down_scale=8,
                                 weighted_sum=False)
            self.assertEqual(spm.speech_encoder_layer, share_layers_pair[1])
            self.assertEqual(spm.nlp_encoder_layer, 6)
            self.assertEqual(len(spm.list_no_grad), 0)

    def test_weight_sum(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        spm = HFSpeechMixEED('facebook/wav2vec2-base-960h', "voidful/bart-base-chinese",
                             fixed_parameters=False, share_layer_ratio=0, down_scale=8,
                             weighted_sum=True)
        result = spm([torch.tensor(ds[0]["audio"]["array"], device=spm.device)],
                     labels=spm.tokenizer.encode(ds[0]['text'], return_tensors='pt').to(spm.device),
                     return_model_detail=True)
        self.assertEqual(result['weighted_sum'].shape[0], 13)

    def test_downscale(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        down_scales = [1, 2, 4, 8]
        for down_scale in down_scales:
            spm = HFSpeechMixEED('facebook/wav2vec2-base-960h', "voidful/bart-base-chinese",
                                 fixed_parameters=False, share_layer_ratio=0.5, down_scale=down_scale,
                                 weighted_sum=False)
            result = spm([torch.tensor(ds[0]["audio"]["array"], device=spm.device)],
                         labels=spm.tokenizer.encode(ds[0]['text'], return_tensors='pt').to(spm.device),
                         return_model_detail=True)
            self.assertAlmostEqual(
                round(result['shape_before_length_adapter'][1] / result['shape_before_enc_dec_projector'][1]),
                down_scale, 1)

        self.assertEqual(spm.speech_encoder_layer, 6)
        self.assertEqual(spm.nlp_encoder_layer, 6)
        self.assertEqual(len(spm.list_no_grad), 0)


if __name__ == '__main__':
    unittest.main()
