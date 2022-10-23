from torch.utils.data import Dataset
import torch


class MyDataSet(Dataset):
    def __init__(self, json_dict, train, transform):
        super(MyDataSet, self).__init__()
        self.json_dict = json_dict
        self.train = train
        # 分词器和词典
        self.tokenizer = transform
        # 处理训练数据集和测试数据集

    def __len__(self):
        return len(self.json_dict)

    def __getitem__(self, index):
        pid = self.json_dict[index]['id']
        title_encoding = self.tokenizer(self.json_dict[index]['title'], truncation=True, padding='max_length',
                                        max_length=16)
        assignee_encoding = self.tokenizer(self.json_dict[index]['assignee'], truncation=True, padding='max_length',
                                           max_length=16)
        abstract_encoding = self.tokenizer(self.json_dict[index]['abstract'], truncation=True, padding='max_length',
                                           max_length=64)
        label = torch.tensor(int(self.json_dict[index]['label_id']))
        id_encoding = {"id": pid}
        title_encoding = {key: torch.tensor(val) for key, val in title_encoding.items()}
        assignee_encoding = {key: torch.tensor(val) for key, val in assignee_encoding.items()}
        abstract_encoding = {key: torch.tensor(val) for key, val in abstract_encoding.items()}
        return [id_encoding, title_encoding, assignee_encoding, abstract_encoding], label
