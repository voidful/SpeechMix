import unittest
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from train import create_self_decoder_input


class TestModel(unittest.TestCase):
    def test_t5_create_self_decoder_input(self):
        test_sent = 'the ordinary duties of life mister daly anxious to make some return for the kindness shown him offered to act as tutor to all the children who were old enough for school duties'
        for config in ["valhalla/t5-small-qg-hl", 'facebook/bart-base']:
            tokenizer = AutoTokenizer.from_pretrained(config)
            model = AutoModelForSeq2SeqLM.from_pretrained(config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            i, t = create_self_decoder_input(model.to(device), tokenizer, test_sent, device)
            gen_result = model.generate(torch.tensor([i], device=device), num_beams=1).tolist()
            print("input:", i)
            print("input tokenize:", tokenizer.batch_decode([i]))
            print("target:", t, )
            print("target tokenize:", tokenizer.batch_decode([t]))
            print('generate:', gen_result)
            print("generate tokenize:", tokenizer.batch_decode(gen_result))

            loss_a = model(input_ids=torch.tensor([i], device=device),
                           labels=torch.tensor([t], device=device)).loss.data.tolist()
            loss_b = model(input_ids=torch.tensor([i], device=device),
                           labels=torch.tensor(gen_result, device=device)).loss.data.tolist()
            print(loss_a, loss_b)
            self.assertTrue(loss_a < loss_b)


if __name__ == '__main__':
    unittest.main()
