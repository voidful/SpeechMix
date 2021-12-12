import math

from torch import nn
from transformers import Wav2Vec2Model, AutoModelForSeq2SeqLM, SpeechEncoderDecoderModel, AutoTokenizer, \
    Wav2Vec2FeatureExtractor, HubertModel, UniSpeechSatModel
import torch


def handle_decoder_input_none(decoder_config, batch=1, device='cpu'):
    return torch.tensor([[decoder_config.decoder_start_token_id]] * batch).to(device)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class SpeechMixED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, ftl=False):
        super(SpeechMixED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(speech_model_config, nlp_model_config)
        self.model.config.decoder_start_token_id = self.model.config.decoder.decoder_start_token_id
        self.model.config.pad_token_id = self.model.config.decoder.pad_token_id
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        if ftl:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any([k in name for k in ["encoder_attn", "enc_to_dec_proj", "embed_tokens"]]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    def forward(self, input_values, attention_mask=None, decoder_input_ids=None, labels=None):
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(self.model.config.decoder, device=self.device)
        outputs = self.model(input_values=input_values, attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids, labels=labels)
        return_dict = {'logits': torch.argmax(outputs['logits'], -1)}
        if 'loss' in outputs:
            return_dict['loss'] = outputs['loss']
        return return_dict


class SpeechMixEED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, lnae=False, lna=False, fne=True, remove_layers=None):
        super(SpeechMixEED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'hubert' in speech_model_config:
            self.encoder_model = HubertModel.from_pretrained(speech_model_config)
        elif 'unispeech' in speech_model_config:
            self.encoder_model = UniSpeechSatModel.from_pretrained(speech_model_config)
        else:
            self.encoder_model = Wav2Vec2Model.from_pretrained(speech_model_config).to(self.device)
        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        num_nlp_encoder_layers = 0
        if hasattr(self.decoder_model.base_model.encoder, 'layers'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.layers)
        elif hasattr(self.decoder_model.base_model.encoder, 'block'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.block)

        # remove last x layer
        if remove_layers is None:
            if num_nlp_encoder_layers == 0 or num_nlp_encoder_layers <= len(self.encoder_model.encoder.layers):
                remove_layers = int(len(self.encoder_model.encoder.layers) * 1 / 3)
            else:
                remove_layers = len(self.encoder_model.encoder.layers)
        self.encoder_model.encoder.layers = self.encoder_model.encoder.layers[
                                            :len(self.encoder_model.encoder.layers) - remove_layers]
        self.num_speech_encoder_layers = len(self.encoder_model.encoder.layers)
        self.enc_to_dec_proj = nn.Linear(self.encoder_model.config.hidden_size,
                                         self.decoder_model.config.hidden_size).to(self.device)
        self.downsize = 4
        self.downloop = int(math.log(self.downsize, 2))
        self.length_adapters = nn.ModuleList([nn.Conv1d(in_channels=self.encoder_model.config.hidden_size,
                                                        out_channels=self.encoder_model.config.hidden_size,
                                                        kernel_size=2,
                                                        stride=2).to(self.device)] * self.downloop)
        self.custom_modules()
        if lna or lnae:
            for xcoder in [self.encoder_model.named_parameters, self.decoder_model.named_parameters]:
                for name, param in xcoder():
                    if param.requires_grad:
                        if any([k in name for k in
                                ["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                                 "layernorm_embedding"]]) or (
                                lnae and ('attention' in name or ('encoder' in name))):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
        elif fne:
            for name, param in self.encoder_model.named_parameters():
                if param.requires_grad:
                    if 'feature_extractor' in name:
                        param.requires_grad = False
            for name, param in self.decoder_model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
        for name, p in self.named_parameters():
            print(name, p.requires_grad)

    def custom_modules(self):
        return None

    def cal_loss(self, inputs_embeds=None, text_input_ids=None, attention_mask=None, decoder_input_ids=None,
                 labels=None):
        if inputs_embeds is not None:
            output = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids, labels=labels)
        elif text_input_ids is not None:
            output = self.decoder_model(input_ids=text_input_ids,
                                        decoder_input_ids=decoder_input_ids, labels=labels)
        return output

    def forward(self, input_values, text_input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(self.decoder_model.config, input_values.shape[0],
                                                          device=self.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.decoder_model.config.pad_token_id,
                                                   self.decoder_model.config.decoder_start_token_id)
        encoder_outputs = self.encoder_model(input_values=input_values, attention_mask=attention_mask)
        inputs_embeds = encoder_outputs['last_hidden_state']
        for loop in range(self.downloop):
            inputs_embeds = self.length_adapters[loop](inputs_embeds.transpose(1, 2)).transpose(1, 2)
        inputs_embeds = self.enc_to_dec_proj(inputs_embeds)

        downsamp_mask = None
        if attention_mask is not None:
            downsamp_mask = self.encoder_model._get_feature_vector_attention_mask(
                encoder_outputs['extract_features'].shape[1], attention_mask)
            downsamp_mask = (torch.sum(downsamp_mask, dim=-1) / self.downsize).int().tolist()
            downsamp_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([1] * i, device=self.device) for i in downsamp_mask],
                batch_first=True)
            pad_length = downsamp_mask.shape[1]
            repad_length = inputs_embeds.shape[1]
            downsamp_mask = torch.nn.ConstantPad1d((0, repad_length - pad_length), 0)(downsamp_mask)
        outputs = self.cal_loss(inputs_embeds=inputs_embeds, text_input_ids=text_input_ids,
                                attention_mask=downsamp_mask,
                                decoder_input_ids=decoder_input_ids, labels=labels)
        return_dict = {'logits': torch.argmax(outputs['logits'], -1)}
        if 'loss' in outputs:
            return_dict['loss'] = outputs['loss']
        return return_dict


class SpeechMixSelf(SpeechMixEED):

    def cal_loss(self, inputs_embeds=None, text_input_ids=None, attention_mask=None, decoder_input_ids=None,
                 labels=None):
        if labels is not None:
            labels = labels.to(self.device)
        outputs = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     output_hidden_states=True,
                                     decoder_input_ids=decoder_input_ids, labels=labels)
        if labels is not None:
            nlp_outputs = self.decoder_model(input_ids=text_input_ids, output_hidden_states=True,
                                             decoder_input_ids=decoder_input_ids, labels=labels)
            nlp_hidden = nlp_outputs['encoder_hidden_states'][-1]
            speech_hidden = outputs['encoder_hidden_states'][-1]
            attn_output = torch.bmm(nlp_hidden,
                                    speech_hidden.view(nlp_hidden.shape[0], self.decoder_model.config.hidden_size, -1))
            voice_projected_embeds = torch.bmm(attn_output, speech_hidden)

            mse_loss_fn = torch.nn.MSELoss()
            kld_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            kld_loss = kld_loss_fn(torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                                   torch.nn.functional.softmax(nlp_outputs.logits, dim=-1))
            mse_loss = mse_loss_fn(voice_projected_embeds, nlp_hidden)
            # mse_loss + kld_loss +
            loss = outputs.loss + mse_loss
            # + outputs.loss
            outputs['mse_loss'] = mse_loss.mean().item()
            outputs['kld_loss'] = kld_loss.mean().item()
            outputs['ce_loss'] = outputs.loss.mean().item()
            # print(outputs['mse_loss'], outputs['kld_loss'], outputs['ce_loss'])
            outputs['loss'] = loss.mean()

        return outputs


class SpeechMixAdapt(SpeechMixEED):

    def custom_modules(self):
        if hasattr(self.decoder_model.base_model.encoder, 'layers'):
            decoder_stack = [self.decoder_model.base_model.encoder.layers,
                             self.decoder_model.base_model.decoder.layers]
        elif hasattr(self.decoder_model.base_model.encoder, 'block'):
            decoder_stack = [self.decoder_model.base_model.encoder.block,
                             self.decoder_model.base_model.decoder.block]

        for encoder_parameter in decoder_stack:
            for name, param in encoder_parameter.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
        for decoder_parameter in decoder_stack:
            for name, param in decoder_parameter.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False

        embshape = self.decoder_model.config.d_model
        bottleneck = int(embshape / 2)
        self.adapters = nn.ModuleList()
        [[self.adapters.append(nn.Sequential(nn.LayerNorm(embshape), nn.Linear(embshape, bottleneck), nn.ReLU(),
                                             nn.Linear(bottleneck, embshape))) for _ in model_decoder_layers] for
         model_decoder_layers in decoder_stack]

        for s_i, s in enumerate(decoder_stack):
            for l_i, l in enumerate(s):
                l.register_forward_hook(lambda m, i, o: (self.adapters[s_i * len(s) + l_i](o[0]), o[1:]))


class SpeechMixGAN(SpeechMixEED):

    def custom_modules(self):
        self.discriminator = nn.Linear(self.decoder_model.config.hidden_size ** 2, 1).to(self.device)
        self.des_update = 1000
        self.update_count = 1
        self.keep_update = 1000
        return None

    def cal_loss(self, inputs_embeds=None, text_input_ids=None, attention_mask=None, decoder_input_ids=None,
                 labels=None):
        outputs = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     output_hidden_states=True,
                                     decoder_input_ids=decoder_input_ids)
        loss = 0
        if labels is not None:
            if self.training:
                if self.update_count % self.des_update == 0:
                    if self.keep_update > 0:
                        self.keep_update -= 1
                        for name, p in self.named_parameters():
                            if 'discriminator' in name:
                                p.grad = None
                    else:
                        self.keep_update = 1000
                        self.update_count += 1
                else:
                    self.update_count += 1
                    for name, p in self.named_parameters():
                        if 'discriminator' not in name:
                            p.grad = None

            labels = labels.to(self.device)
            nlp_outputs = self.decoder_model(labels, output_hidden_states=True,
                                             decoder_input_ids=decoder_input_ids)

            voice_hidden = outputs['decoder_hidden_states'][-1]
            nlp_hidden = nlp_outputs['decoder_hidden_states'][-1]
            nlp_encoder_hidden = nlp_outputs['encoder_hidden_states'][-1]

            loss_fn = torch.nn.BCEWithLogitsLoss()
            voice_enc_attn_output = torch.bmm(
                inputs_embeds.view(inputs_embeds.shape[0], self.decoder_model.config.hidden_size, -1),
                inputs_embeds.view(inputs_embeds.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)
            vt_enc_loss = loss_fn(self.discriminator(voice_enc_attn_output).flatten(),
                                  torch.ones(voice_enc_attn_output.shape[0]).to(self.device))

            nlp_enc_attn_output = torch.bmm(
                nlp_encoder_hidden.view(nlp_encoder_hidden.shape[0], self.decoder_model.config.hidden_size, -1),
                nlp_encoder_hidden.view(nlp_encoder_hidden.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)

            nt_enc_loss = loss_fn(self.discriminator(nlp_enc_attn_output).flatten(),
                                  torch.zeros(nlp_enc_attn_output.shape[0]).to(self.device))

            voice_attn_output = torch.bmm(
                voice_hidden.view(voice_hidden.shape[0], self.decoder_model.config.hidden_size, -1),
                voice_hidden.view(voice_hidden.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)

            vt_loss = loss_fn(self.discriminator(voice_attn_output).flatten(),
                              torch.ones(voice_attn_output.shape[0]).to(self.device))

            nlp_attn_output = torch.bmm(
                nlp_hidden.view(nlp_hidden.shape[0], self.decoder_model.config.hidden_size, -1),
                nlp_hidden.view(nlp_hidden.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)

            nt_loss = loss_fn(self.discriminator(nlp_attn_output).flatten(),
                              torch.zeros(nlp_attn_output.shape[0]).to(self.device))

            loss += vt_loss + nt_loss + nt_enc_loss + vt_enc_loss
        outputs['loss'] = loss
        return outputs
