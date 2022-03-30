import math

from torch import nn
from transformers import AutoModelForSeq2SeqLM, SpeechEncoderDecoderModel, AutoTokenizer, \
    Wav2Vec2FeatureExtractor
import torch
import s3prl.hub as hub
import torch.nn.functional as F


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
    def __init__(self, speech_model_config, nlp_model_config, fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], **kwargs):
        super(SpeechMixED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(speech_model_config, nlp_model_config)
        self.model.config.decoder_start_token_id = self.model.config.decoder.decoder_start_token_id
        self.model.config.pad_token_id = self.model.config.decoder.pad_token_id
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        self.model.freeze_feature_encoder()
        if fixed_parameters:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any([k in name for k in fixed_except]):
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
    def __init__(self, speech_model_config, nlp_model_config, share_layer_ratio=0,
                 down_scale=8, weighted_sum=False,
                 fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], **kwargs):
        super(SpeechMixEED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_model = getattr(hub, speech_model_config)().to(self.device)
        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        self.weighted_sum = weighted_sum

        num_nlp_encoder_layers = 0
        if hasattr(self.decoder_model.base_model.encoder, 'layers'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.layers)
        elif hasattr(self.decoder_model.base_model.encoder, 'block'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.block)

        print("Before layer sharing num_speech_encoder_layers", len(self.encoder_model.model.encoder.layers))
        remove_layers = int(
            len(self.encoder_model.model.encoder.layers) * share_layer_ratio) if share_layer_ratio != 0 else 0
        self.encoder_model.model.encoder.layers = self.encoder_model.model.encoder.layers[
                                                  :len(self.encoder_model.model.encoder.layers) - remove_layers]
        self.num_speech_encoder_layers = len(self.encoder_model.model.encoder.layers)
        print("After layer sharing ",
              "num_speech_encoder_layers", len(self.encoder_model.model.encoder.layers),
              "num_nlp_encoder_layers", num_nlp_encoder_layers,
              "share_layer_ratio", share_layer_ratio,
              "remove_layers", remove_layers, )

        # Downsample
        self.downsize = down_scale
        self.downloop = int(math.log(self.downsize, 2))
        if self.downsize > 1:
            self.length_adapters = nn.Sequential(
                *[nn.Conv1d(in_channels=self.encoder_model.model.final_proj.in_features,
                            out_channels=self.encoder_model.model.final_proj.in_features,
                            kernel_size=2,
                            stride=2) for _ in range(self.downloop)]).to(self.device)
        else:
            self.length_adapters = nn.Sequential(nn.Identity()).to(self.device)

        self.weights_sum = nn.Parameter(torch.zeros(self.num_speech_encoder_layers)).to(self.device)
        self.enc_to_dec_proj = nn.Linear(self.encoder_model.model.final_proj.in_features,
                                         self.decoder_model.config.hidden_size).to(self.device)
        self.custom_modules(**kwargs)
        if fixed_parameters:
            self.encoder_model.eval()
            self.decoder_model.eval()
            for xcoder in [self.encoder_model.named_parameters, self.decoder_model.named_parameters]:
                for name, param in xcoder():
                    if param.requires_grad:
                        if any([k in name for k in fixed_except]):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

        list_no_grad = []
        list_grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                list_grad.append(name)
            else:
                list_no_grad.append(name)

        self.speech_encoder_layer = len(self.encoder_model.model.encoder.layers)
        self.nlp_emb = self.decoder_model.get_input_embeddings()
        self.nlp_encoder_layer = num_nlp_encoder_layers
        self.list_grad = list_grad
        self.list_no_grad = list_no_grad

    def custom_modules(self, **kwargs):
        return None

    def cal_loss(self, inputs_embeds=None,  attention_mask=None, decoder_input_ids=None,
                 labels=None):
        if inputs_embeds is not None:
            output = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids, labels=labels)
        return output

    def forward(self, input_values, input_text_prompt=None,decoder_input_ids=None, labels=None,
                return_model_detail=False):
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(self.decoder_model.config, len(input_values),
                                                          device=self.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.decoder_model.config.pad_token_id,
                                                   self.decoder_model.config.decoder_start_token_id)
        return_dict = {}
        encoder_outputs = self.encoder_model(input_values)
        inputs_embeds = encoder_outputs['last_hidden_state'].to(self.device)
        if self.weighted_sum:
            # weighted sum
            stacked_feature = torch.stack(encoder_outputs['hidden_states'], dim=0)
            _, *origin_shape = stacked_feature.shape
            stacked_feature = stacked_feature.view(self.num_speech_encoder_layers, -1)
            norm_weights = F.softmax(self.weights_sum, dim=-1)
            if return_model_detail:
                return_dict['weighted_sum'] = norm_weights
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
            inputs_embeds = weighted_feature.view(*origin_shape)
        if return_model_detail:
            return_dict['shape_before_length_adapter'] = inputs_embeds.shape
        inputs_embeds = self.length_adapters(inputs_embeds.transpose(1, 2)).transpose(1, 2)
        if return_model_detail:
            return_dict['shape_before_enc_dec_projector'] = inputs_embeds.shape
        inputs_embeds = self.enc_to_dec_proj(inputs_embeds)
        if return_model_detail:
            return_dict['shape_after_enc_dec_projector'] = inputs_embeds.shape
        if input_text_prompt is not None:
            text_prompt = self.nlp_emb(
                self.tokenizer(input_text_prompt, return_tensors='pt')['input_ids'].to(self.device))
            inputs_embeds = torch.cat((text_prompt, inputs_embeds), 1)
        outputs = self.cal_loss(inputs_embeds=inputs_embeds,
                                decoder_input_ids=decoder_input_ids, labels=labels)
        return_dict['logits'] = torch.argmax(outputs['logits'], -1)
        if 'loss' in outputs:
            return_dict['loss'] = outputs['loss']
        return return_dict


class SpeechMixFixed(SpeechMixEED):

    def custom_modules(self, fixed_speech=False, fixed_nlp=True, **kwargs):
        print(fixed_speech, fixed_nlp, kwargs)
        self.encoder_model.eval()
        self.decoder_model.eval()
        if fixed_speech:
            for name, param in self.encoder_model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
        if fixed_nlp:
            for name, param in self.decoder_model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False


class SpeechMixAdapter(SpeechMixEED):

    def custom_modules(self, **kwargs):
        self.encoder_model.eval()
        self.decoder_model.eval()
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

        embshape = self.decoder_model.config.d_model
        bottleneck = int(embshape / 2)
        self.adapters = nn.ModuleList()
        [[self.adapters.append(nn.Sequential(nn.LayerNorm(embshape), nn.Linear(embshape, bottleneck), nn.ReLU(),
                                             nn.Linear(bottleneck, embshape))) for _ in model_decoder_layers] for
         model_decoder_layers in decoder_stack]
        self.adapters.to(self.device)
        for s_i, s in enumerate(decoder_stack):
            for l_i, l in enumerate(s):
                l.register_forward_hook(lambda m, i, o: (self.adapters[s_i * len(s) + l_i](o[0]), o[1:]))


class SpeechMixSelf(SpeechMixEED):

    def custom_modules(self, **kwargs):
        self.encoder_model.eval()
        self.decoder_model.eval()

        for name, param in self.decoder_model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False

    def cal_loss(self, inputs_embeds=None, text_input_ids=None, attention_mask=None, decoder_input_ids=None,
                 labels=None):
        if labels is not None:
            labels = labels.to(self.device)
        self.decoder_model.eval()
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
            softmax = torch.nn.Softmax(dim=-1)
            attn_output = softmax(attn_output / math.sqrt(self.decoder_model.config.hidden_size))
            voice_projected_embeds = torch.bmm(attn_output, speech_hidden)
            mse_loss_fn = torch.nn.MSELoss()
            mse_loss = mse_loss_fn(voice_projected_embeds, nlp_hidden)

            kld_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            kld_loss = kld_loss_fn(torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                                   torch.nn.functional.softmax(nlp_outputs.logits, dim=-1))
            ce_loss = outputs.loss
            loss = kld_loss + ce_loss + mse_loss

            # print(mse_loss.mean().item(), kld_loss.mean().item(), ce_loss.mean().item())
            outputs['loss'] = loss.mean()

        return outputs


class SpeechMixGAN(SpeechMixEED):

    def custom_modules(self, **kwargs):
        self.encoder_model.eval()
        self.decoder_model.eval()

        for name, param in self.decoder_model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False

        self.discriminator = nn.Linear(self.decoder_model.config.hidden_size ** 2, 1).to(self.device)
        self.des_update = 1000
        self.update_count = 1
        self.keep_update = 1000

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
