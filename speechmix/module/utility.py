from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, TrainerCallback, \
    TrainerState, TrainerControl
from transformers import TrainingArguments


class FreezingCallback(TrainerCallback):
    def __init__(self, trainer, freeze_model, freeze_epoch=3):
        self.trainer = trainer
        self.freeze_model = freeze_model
        self.freeze_epoch = freeze_epoch
        self.current_step_idx = 0
        self.default_param_fix = {}
        self.name_list = []
        for name, param in self.freeze_model.named_parameters():
            self.name_list.append(name)
            self.default_param_fix[name] = param.requires_grad
        self.freeze_layers = int(len(self.default_param_fix.keys()) / freeze_epoch)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch < self.freeze_epoch:
            release = self.name_list[-int(self.freeze_layers * state.epoch):]
            for name, param in self.freeze_model.named_parameters():
                if name in release:
                    param.requires_grad = self.default_param_fix[name]
                else:
                    param.requires_grad = False
        else:
            for name, param in self.freeze_model.named_parameters():
                param.requires_grad = self.default_param_fix[name]
        self.current_step_idx += 1

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for name, param in self.trainer.model.named_parameters():
            param.requires_grad = True
