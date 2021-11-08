from torch import nn
from transformers import Wav2Vec2Model, AutoModelForSeq2SeqLM, SpeechEncoderDecoderModel, AutoTokenizer, \
    Wav2Vec2Processor, HubertModel, UniSpeechSatModel
import torch


def handle_decoder_input_none(decoder_config, device='cpu'):
    return torch.tensor([[decoder_config.decoder_start_token_id]]).to(device)


class SpeechMixED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, ftl=False):
        super(SpeechMixED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(speech_model_config, nlp_model_config)
        self.model.config.decoder_start_token_id = self.model.config.decoder.decoder_start_token_id
        self.model.config.pad_token_id = self.model.config.decoder.pad_token_id
        self.processor = Wav2Vec2Processor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        if ftl:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any([k in name for k in ["encoder_attn", "enc_to_dec_proj", "embed_tokens"]]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    def forward(self, input_values, decoder_input_ids=None, labels=None):
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(self.model.config.decoder, self.device)
        outputs = self.model(input_values=input_values, decoder_input_ids=decoder_input_ids, labels=labels)
        return_dict = {'logits': torch.argmax(outputs['logits'], -1)}
        if outputs.get('loss', None):
            return_dict['loss'] = outputs['loss']
        return return_dict


class SpeechMixEED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, lnae=False, lna=False, fne=True):
        super(SpeechMixEED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'hubert' in speech_model_config:
            self.encoder_model = HubertModel.from_pretrained(speech_model_config)
        elif 'unispeech' in speech_model_config:
            self.encoder_model = UniSpeechSatModel.from_pretrained(speech_model_config)
        else:
            self.encoder_model = Wav2Vec2Model.from_pretrained(speech_model_config).to(self.device)
        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        # remove last x layer
        self.encoder_model.encoder.layers = self.encoder_model.encoder.layers[
                                            :-len(self.decoder_model.model.encoder.layers)]
        self.enc_to_dec_proj = nn.Linear(self.encoder_model.config.hidden_size,
                                         self.decoder_model.config.hidden_size).to(self.device)
        self.length_adapter = nn.Conv1d(self.encoder_model.config.hidden_size, self.encoder_model.config.hidden_size,
                                        2,
                                        stride=2).to(self.device)
        if lna or lnae:
            for xcoder in [self.encoder_model.named_parameters, self.decoder_model.named_parameters]:
                for name, param in xcoder():
                    if param.requires_grad:
                        if any([k in name for k in
                                ["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter', 'lm_head',
                                 'shared']]) or (
                                lnae and ('attention' in name or ('encoder' in name and 'self_attn' in name))):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
        elif fne:
            for name, param in self.decoder_model.named_parameters():
                if param.requires_grad and not any([k in name for k in ['lm_head', 'shared']]):
                    param.requires_grad = False

    def forward(self, input_values, decoder_input_ids=None, labels=None):
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(self.decoder_model.config, self.device)
        inputs_embeds = self.encoder_model(input_values=input_values)['last_hidden_state']
        for _ in range(3):
            inputs_embeds = self.length_adapter(inputs_embeds.transpose(1, 2)).transpose(1, 2)
        projected_embeds = self.enc_to_dec_proj(inputs_embeds)
        outputs = self.decoder_model(inputs_embeds=projected_embeds,
                                     decoder_input_ids=decoder_input_ids, labels=labels)
        return_dict = {'logits': torch.argmax(outputs['logits'], -1)}
        if outputs.get('loss', None):
            return_dict['loss'] = outputs['loss']
        return return_dict


class SpeechMixSelf(SpeechMixEED):

    def forward(self, input_values, input_ids=None, decoder_input_ids=None, labels=None):
        if decoder_input_ids is None:
            decoder_input_ids = handle_decoder_input_none(self.decoder_model.config, self.device)
        inputs_embeds = self.encoder_model(input_values=input_values.to(self.device))['last_hidden_state']
        for _ in range(3):
            inputs_embeds = self.length_adapter(inputs_embeds.transpose(1, 2)).transpose(1, 2)
        projected_embeds = self.enc_to_dec_proj(inputs_embeds)
        outputs = self.decoder_model(inputs_embeds=projected_embeds,
                                     decoder_input_ids=decoder_input_ids)
        loss = None
        if input_ids is not None:
            nlp_hidden = \
                self.decoder_model(input_ids.to(self.device), output_hidden_states=True)['encoder_hidden_states'][0]
            attn_output = torch.bmm(nlp_hidden,
                                    projected_embeds.view(nlp_hidden.shape[0], self.decoder_model.config.hidden_size,
                                                          -1))
            voice_projected_embeds = torch.bmm(attn_output, projected_embeds)

            kld_outputs = self.decoder_model(input_ids.to(self.device),
                                             decoder_input_ids=decoder_input_ids)

            mse_loss_fn = torch.nn.MSELoss()
            kld_loss_fn = torch.nn.KLDivLoss()

            loss = kld_loss_fn(kld_outputs.logits, outputs.logits)
            loss += mse_loss_fn(voice_projected_embeds, nlp_hidden)

        return_dict = {'logits': torch.argmax(outputs['logits'], -1)}
        if loss is not None:
            return_dict['loss'] = loss
        return return_dict
