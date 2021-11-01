from torch import nn
from transformers import Wav2Vec2Model, AutoModelForSeq2SeqLM, SpeechEncoderDecoderModel, AutoTokenizer, \
    Wav2Vec2Processor
import torch


def handle_decoder_input_none(decoder_config):
    return torch.tensor([[decoder_config.decoder_start_token_id]])


class SpeechMixED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, ftl=False):
        super(SpeechMixED, self).__init__()
        self.model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(speech_model_config, nlp_model_config)
        self.processor = Wav2Vec2Processor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        if ftl:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any([k in name for k in ["encoder_attn", "enc_to_dec_proj", "embed_tokens"]]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    def forward(self, input_values, decoder_input_ids=None):
        if decoder_input_ids is None:
            decoder_input_ids = handle_decoder_input_none(self.model.config.decoder)
        outputs = self.model(input_values, decoder_input_ids=decoder_input_ids)
        return outputs


class SpeechMixEED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, lna=False, fne=False):
        super(SpeechMixEED, self).__init__()
        self.encoder_model = Wav2Vec2Model.from_pretrained(speech_model_config)
        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config)
        self.processor = Wav2Vec2Processor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        # remove last x layer
        self.encoder_model.encoder.layers = self.encoder_model.encoder.layers[
                                            :len(self.decoder_model.model.encoder.layers)]
        self.enc_to_dec_proj = nn.Linear(self.encoder_model.config.hidden_size, self.decoder_model.config.hidden_size)
        self.length_adapter = nn.Conv1d(self.encoder_model.config.hidden_size, self.encoder_model.config.hidden_size,
                                        2,
                                        stride=2)
        if lna:
            for xcoder in [self.encoder_model.named_parameters, self.decoder_model.named_parameters]:
                for name, param in xcoder():
                    if param.requires_grad:
                        if any([k in name for k in ["layer_norm", "attention"]]):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
        elif fne:
            for name, param in self.decoder_model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False

    def forward(self, input_values, decoder_input_ids=None):
        if decoder_input_ids is None:
            decoder_input_ids = handle_decoder_input_none(self.decoder_model.config)
        inputs_embeds = self.encoder_model(input_values=input_values)['last_hidden_state']
        for _ in range(3):
            inputs_embeds = self.length_adapter(inputs_embeds.transpose(1, 2)).transpose(1, 2)
        projected_embeds = self.enc_to_dec_proj(inputs_embeds)
        outputs = self.decoder_model(inputs_embeds=projected_embeds,
                                     decoder_input_ids=decoder_input_ids)
        return outputs
