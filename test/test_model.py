import unittest
import pytest
from datasets import load_dataset
import torch
from speechmix import SpeechMixEED, SpeechMixAdapter, SpeechMixSelf, SpeechMixGAN
from train import create_self_decoder_input


class TestModel(unittest.TestCase):
    def test_pure(self):
        spm = SpeechMixEED('wav2vec2', "voidful/bart-base-chinese",
                           fixed_parameters=False, share_layer_ratio=0, down_scale=8,
                           weighted_sum=False)
        self.assertEqual(spm.speech_encoder_layer, 12)
        self.assertEqual(spm.nlp_encoder_layer, 6)
        self.assertEqual(len(spm.list_no_grad), 0)

    def test_share_layer(self):
        for share_layers_pair in [(1, 0), (0.5, 6), (0, 12)]:
            spm = SpeechMixEED('wav2vec2', "voidful/bart-base-chinese",
                               fixed_parameters=False, share_layer_ratio=share_layers_pair[0], down_scale=8,
                               weighted_sum=False)
            self.assertEqual(spm.speech_encoder_layer, share_layers_pair[1])
            self.assertEqual(spm.nlp_encoder_layer, 6)
            self.assertEqual(len(spm.list_no_grad), 0)

    def test_weight_sum(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        spm = SpeechMixEED('wav2vec2', "voidful/bart-base-chinese",
                           fixed_parameters=False, share_layer_ratio=0, down_scale=8,
                           weighted_sum=True)
        result = spm([torch.tensor(ds[0]["audio"]["array"], device=spm.device)],
                     labels=spm.tokenizer.encode(ds[0]['text'], return_tensors='pt').to(spm.device),
                     return_model_detail=True)
        self.assertEqual(result['weighted_sum'].shape[0], 12)

    def test_downscale(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        down_scales = [1, 2, 4, 8]
        for down_scale in down_scales:
            spm = SpeechMixEED('wav2vec2', "voidful/bart-base-chinese",
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

    def test_adapter(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        spm = SpeechMixAdapter('wav2vec2', "voidful/bart-base-chinese",
                               fixed_parameters=False, share_layer_ratio=0.4, down_scale=8, )
        result = spm([torch.tensor(ds[0]["audio"]["array"], device=spm.device)],
                     labels=spm.tokenizer.encode(ds[0]['text'], return_tensors='pt').to(spm.device),
                     return_model_detail=True)
        self.assertEqual(spm.speech_encoder_layer, 8)
        self.assertEqual(spm.nlp_encoder_layer, 6)

    def test_self(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        spm = SpeechMixSelf('wav2vec2', "voidful/bart-base-chinese",
                            fixed_parameters=False, share_layer_ratio=0.4, down_scale=8, )
        i, t = create_self_decoder_input(spm.decoder_model.to(spm.device), spm.tokenizer, ds[0]['text'], spm.device)
        result = spm([torch.tensor(ds[0]["audio"]["array"], device=spm.device)],
                     text_input_ids=torch.tensor([i], device=spm.device),
                     labels=torch.tensor([t], device=spm.device),
                     return_model_detail=True)
        print(result)
        self.assertEqual(spm.speech_encoder_layer, 8)
        self.assertEqual(spm.nlp_encoder_layer, 6)

    def test_gan(self):
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        spm = SpeechMixGAN('wav2vec2', "voidful/bart-base-chinese",
                           fixed_parameters=False, share_layer_ratio=0.4, down_scale=8, )
        result = spm([torch.tensor(ds[0]["audio"]["array"], device=spm.device)],
                     labels=spm.tokenizer.encode(ds[0]['text'], return_tensors='pt').to(spm.device),
                     return_model_detail=True)
        print(result)
        self.assertEqual(spm.speech_encoder_layer, 8)
        self.assertEqual(spm.nlp_encoder_layer, 6)


if __name__ == '__main__':
    unittest.main()
