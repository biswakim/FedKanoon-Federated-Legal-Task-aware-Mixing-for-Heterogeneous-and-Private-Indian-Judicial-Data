"""One model per task; the *only* changes from text classification are the output and loss."""
import torch
from torch import nn
import torch.nn.functional as F

class TaskModel(nn.Module):
    def __init__(self, checkpoint, task, num_labels, use_lora=True, toy=False):
        super().__init__()
        self.task = task
        self.toy = toy
        if toy:
            self.encoder = nn.Embedding(1000, 32)
            hidden = 32
        else:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(checkpoint)
            hidden = self.encoder.config.hidden_size
            if use_lora:
                from peft import LoraConfig, get_peft_model, TaskType
                cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                                 lora_dropout=0.05, target_modules=['query','value'])
                self.encoder = get_peft_model(self.encoder, cfg)
        self.head = nn.Linear(hidden, 1 if task == 'duration' else num_labels)

    def forward(self, input_ids, attention_mask):
        if self.toy:
            h = self.encoder(input_ids)
        else:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)

    def loss(self, out, target, class_weight=None):
        if self.task == 'duration':
            return F.huber_loss(out.squeeze(-1), target.float(), delta=1.0)
        return F.cross_entropy(out, target.long(), weight=class_weight)

    def trainable_state(self):
        allowed = {n for n,p in self.named_parameters() if p.requires_grad}
        return {n: p.detach().cpu().clone() for n,p in self.named_parameters() if n in allowed}

    def load_trainable(self, weights):
        named = dict(self.named_parameters())
        with torch.no_grad():
            for n,v in weights.items():
                named[n].copy_(v.to(named[n].device, dtype=named[n].dtype))
