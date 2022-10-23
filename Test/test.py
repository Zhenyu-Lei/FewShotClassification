from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
from DataSet import MyDataSet
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np


def tokenTest():
    tokenizer = BertTokenizer.from_pretrained('../bert')
    text = "#游戏##电子竞技##英雄联盟##S12##RNG#10月22日比赛预告\
    5:00 RNG vs T1  RNG直面宿命之敌T1!RNG能否再续Msi荣光，跨越T1，晋级半决赛？Xiaohu中路再遇老对手Faker，谁能对线打出优势，带队挺进四强？\
    S12观赛狂欢今日大奖：【PS5】\
    观赛累计打卡进度，【必得】LOL限定炫彩、游戏内代币等好礼！\
    明日大奖预告： 【安德斯特电竞桌椅套装】\
    "
    myText = tokenizer(text, truncation=True, padding='max_length', max_length=16)

    for x in myText:
        print(x)
    model = BertForSequenceClassification.from_pretrained('../bert', num_labels=36)
    input_ids = torch.tensor([myText['input_ids']])
    attention_mask = torch.tensor([myText['attention_mask']])
    print(input_ids.size())
    labels = torch.tensor([[0]])
    '''
    SequenceClassifierOutput(loss=tensor(3.2435, grad_fn=<NllLossBackward0>), logits=tensor([[ 0.4875,  0.0676,  0.2793,  0.1414,  0.2610,  0.1077, -0.1934, -0.3248,
          0.3681, -0.4899,  0.3789, -0.3018,  0.8700, -0.3241,  0.2872, -0.3503,
         -0.3131,  0.2675, -0.0884, -0.6457, -0.0882,  0.2603,  0.3034, -0.3218,
          0.4225,  0.1909, -1.1907,  0.9696,  0.2398,  0.0395,  0.0981, -0.0410,
          0.2665, -0.0548, -0.1634,  0.8172]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
    '''
    print(input_ids, attention_mask, labels)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(outputs)


def testData():
    json_dict = []
    # 读取json文件,分成训练集和测试集
    for line in open("../Data/train.json", 'r', encoding="UTF8"):
        json_dict.append(json.loads(line))
    text_train, text_test = train_test_split(json_dict, test_size=0.2)

    tokenizer = BertTokenizer.from_pretrained('../bert')
    # 制作dataLoader
    myDataSetTrain = MyDataSet(text_train, True, tokenizer)
    myDataSetTest = MyDataSet(text_test, False, tokenizer)

    # 没啥好说的，暂时batch只能是1
    batch_size = 16
    train = DataLoader(myDataSetTrain, batch_size=batch_size, shuffle=True)
    print(len(train))

    train, label = next(iter(train))
    '''
    {id:[]}
    [title]{
        input_ids:[batch_size*[]],
        token_type_ids:[batch_size*[]],
        attention_mask:[batch_size*[]]
    }
    [assignee]{
        input_ids:[batch_size*[]],
        token_type_ids:[batch_size*[]],
        attention_mask:[batch_size*[]]
    }
    [abstract]{
        input_ids:[batch_size*[]],
        token_type_ids:[batch_size*[]],
        attention_mask:[batch_size*[]]
    }
    '''
    # # 输出每个样本
    # for x in range(batch_size):
    #     print("id:", train[0]['id'][x])
    #     print("title:\n", train[1]['input_ids'][x], train[1]['token_type_ids'][x], train[1]['attention_mask'][x])
    #     print("assignee:\n", train[2]['input_ids'][x], train[2]['token_type_ids'][x], train[2]['attention_mask'][x])
    #     print("abstract:\n", train[3]['input_ids'][x], train[3]['token_type_ids'][x], train[3]['attention_mask'][x])
    # model = BertForSequenceClassification.from_pretrained('../bert', num_labels=36)
    # print(train[1]['input_ids'], train[1]['attention_mask'], label)
    # output = model(train[1]['input_ids'], train[1]['attention_mask'], labels=label)
    # print(output)


def mulTest():
    a = torch.rand((16, 36))
    b = torch.rand((16, 36))
    c = torch.rand((16, 36))
    d = torch.rand((3,))
    # print(a)
    res = a * d[0] + b * d[1] + c * d[2]
    print(np.array(res).shape)


if __name__ == '__main__':
    # tokenTest()
    testData()
    # mulTest()
