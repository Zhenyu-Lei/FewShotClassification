import json

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
# transformers bert相关的模型使用和加载
from transformers import BertTokenizer

from DataSet import MyDataSet
from Model.TripleBert import TripleBert

# 分词器
tokenizer = BertTokenizer.from_pretrained('bert')

json_dict = []
# 读取json文件,分成训练集和测试集
for line in open("./Data/train.json", 'r', encoding="UTF8"):
    json_dict.append(json.loads(line))
text_train, text_test = train_test_split(json_dict, test_size=0.2)

# 制作dataLoader
myDataSetTrain = MyDataSet(text_train, True, tokenizer)
myDataSetTest = MyDataSet(text_test, False, tokenizer)

batch_size = 16
train_loader = DataLoader(myDataSetTrain, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(myDataSetTest, batch_size=batch_size, shuffle=True)

# 由于这里是文本分类任务，所以直接使用BertForSequenceClassification完成加载即可，这里需要制定对应的类别数量。
model1 = TripleBert(num_labels=36)  # 36类样本

try:
    f = open("./model.pth")
    model1.load_state_dict(torch.load("./model.pth"))
except FileNotFoundError:
    print("预训练文件不存在")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

# 优化方法
optimize = AdamW(model1.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1
# 学习率预热, num_warmup_steps：初始预热步数, num_training_steps：整个训练过程的总步数
scheduler = get_linear_schedule_with_warmup(optimize, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss()


# 训练函数
def train():
    model1.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch, labels in train_loader:
        # 正向传播
        title = batch[1]
        assignee = batch[2]
        abstract = batch[3]
        labels = labels.to(device)
        outputs = model1(title, assignee, abstract, labels, device)
        loss = loss_fn(outputs, labels)
        total_train_loss += loss

        # 反向梯度信息
        optimize.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)

        # 参数更新
        optimize.step()
        scheduler.step()

        iter_num += 1
        if iter_num % 10 == 0:
            print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))


def validation():
    model1.eval()
    total_eval_loss = 0
    accuracy = 0
    for batch, labels in test_loader:
        with torch.no_grad():
            # 正常传播
            title = batch[1]
            assignee = batch[2]
            abstract = batch[3]
            labels = labels.to(device)
            outputs = model1(title, assignee, abstract, labels, device)

        loss = loss_fn(outputs, labels)

        total_eval_loss += loss.item()
        accuracy = (outputs.argmax(1) == labels).type(torch.float).sum().item()
        accuracy /= len(test_loader)

    print("Accuracy: %.4f" % accuracy)
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_loader)))
    print("-------------------------------")


for epoch in range(4):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()

torch.save(model1.state_dict(), "model.pth")
