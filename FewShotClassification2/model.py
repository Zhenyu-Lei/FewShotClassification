import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

'''
cls:classification模型
'''


class clsModel(nn.Module):
    def __init__(self, args):
        super(clsModel, self).__init__()
        # 加载bert预训练模型
        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
        # config = BertConfig(output_hidden_states=True)
        # self.bert = BertModel(config=config)
        self.cls = nn.Linear(768 * 4, 36)
        '''
        Text_embedding有三种，word_embeddings，position_embeddings以及token_type_embeddings
        例如input_text[1,64]=>embedding[1,64,768]
        <class 'transformers.models.bert.modeling_bert.BertEmbeddings'> BertEmbeddings(
              (word_embeddings): Embedding(21128, 768, padding_idx=0)
              (position_embeddings): Embedding(512, 768)
              (token_type_embeddings): Embedding(2, 768)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
        )
        '''
        self.text_embedding = self.bert.embeddings
        self.text_cls = nn.Linear(768, 36)

    def build_pre_input(self, data):
        text_inputs = data['text_inputs']
        text_mask = data['text_mask']
        # 将文本进行预处理，进行embedding，返回embedding后的结果以及text_mask
        textembedding = self.text_embedding(text_inputs.cuda(), data['text_type_ids'].cuda())
        return textembedding, text_mask

    def forward(self, data, inference=False, multi=False):
        # 将输入数据预处理，先进行embedding
        inputs_embeds, mask = self.build_pre_input(data)
        # 通过bert产生输出
        bert_out = self.bert(attention_mask=mask, inputs_embeds=inputs_embeds)
        # last 4 mean pooling
        hidden_stats = bert_out.hidden_states[-4:]
        hidden_stats = [i.mean(dim=1) for i in hidden_stats]
        # 将最后四层隐状态抽取后，进行分类
        out = self.cls(torch.cat(hidden_stats, dim=1))
        if inference:
            if multi:
                return out
            else:
                # 选取可能性最高的结果
                return torch.argmax(out, dim=1)
        else:
            all_loss, all_acc, all_pre, label = self.cal_loss(out, data['label'].cuda())
            # 计算loss，accuracy，prediction，label
            return all_loss, all_acc, all_pre, label

    # @staticmethod用于修饰类中方法，在其不创建实例情况下进行调用
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        # 如果tensor运算用不到梯度就将其屏蔽就好
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
