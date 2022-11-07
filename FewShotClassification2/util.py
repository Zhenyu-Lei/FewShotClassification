import logging
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings

warnings.filterwarnings("ignore")
import torch.nn as nn

'''
设置了分层学习率，即让bert学习率小一点儿，分类头学习率大一点儿
调试验证集时分别计算了几种不同评价指标
常用trick： ema swa fgm pgd rdrop f1优化等，目前只用了ema
'''


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


# 设置分层学习率
def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    large_lr = ['']
    optimizer_grouped_parameters = [
        {'params': [j for i, j in model.named_parameters() if
                    (not 'bert' in i and not any(nd in i for nd in no_decay))],
         'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and any(nd in i for nd in no_decay))],
         'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and not any(nd in i for nd in no_decay))],
         'lr': args.bert_learning_rate, 'weight_decay': args.weight_decay},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and any(nd in i for nd in no_decay))],
         'lr': args.bert_learning_rate, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level class ids
    temp_dict = dict()
    no_ignores_2 = list()
    no_ignores_4 = list()
    for key in labels:
        temp_dict[key] = temp_dict.get(key, 0) + 1
    for i in range(36):
        if i in temp_dict.keys():
            if temp_dict[i] > 2:
                no_ignores_2.append(i)
            if temp_dict[i] > 4:
                no_ignores_4.append(i)
    f1_macro = f1_score(labels, predictions, average='macro')  #
    f1_micro = f1_score(labels, predictions, average='micro')  #
    f1_weight = f1_score(labels, predictions, average='weighted')  #
    f1_macro_2 = f1_score(labels, predictions, labels=no_ignores_2, average='macro')  #
    f1_macro_4 = f1_score(labels, predictions, labels=no_ignores_4, average='macro')  #
    eval_results = {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weight': f1_weight, 'f1_macro_2': f1_macro_2,
                    'f1_macro_4': f1_macro_4}
    return eval_results


# FGM
class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}


# PGD
class PGD:
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
