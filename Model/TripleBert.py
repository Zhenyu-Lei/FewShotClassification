from transformers import BertForSequenceClassification
from torch import nn
import torch


class TripleBert(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # 准备3个模型，分别处理title,assignee,abstract
        self.model1 = BertForSequenceClassification.from_pretrained('./bert', num_labels=num_labels)  # 36类样本
        self.model2 = BertForSequenceClassification.from_pretrained('./bert', num_labels=num_labels)  # 36类样本
        self.model3 = BertForSequenceClassification.from_pretrained('./bert', num_labels=num_labels)  # 36类样本
        self.a = torch.rand([3])

    def forward(self, title, assignee, abstract, label):
        vec1 = self.model1(title['input_ids'], title['attention_mask'], labels=label)[1]
        vec2 = self.model2(assignee['input_ids'], assignee['attention_mask'], labels=label)[1]
        vec3 = self.model3(abstract['input_ids'], abstract['attention_mask'], labels=label)[1]
        '''
        logits=tensor([[ 0.7125, -0.5636,  0.2447, -0.2928,  0.0176,  0.5130,  0.2257, -0.0662,
          0.1515, -0.6435, -0.4010, -1.2344,  0.1434, -0.4420,  0.3731,  0.3463,
         -0.5045, -0.6285, -0.3521,  0.5634,  0.0154, -0.7292, -0.1993,  0.2515,
         -0.3855, -0.1398,  0.1992,  0.0995, -0.4023, -0.2450,  0.5737,  0.5621,
          0.1754,  0.3666,  0.3109, -0.2609]], grad_fn=<AddmmBackward0>)
          '''
        return vec1*self.a[0]+vec2*self.a[1]+vec3*self.a[2]
