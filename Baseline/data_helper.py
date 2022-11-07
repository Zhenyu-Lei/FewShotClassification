import json
import random
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import random
from sklearn.model_selection import train_test_split, StratifiedKFold

'''
直接random.shuffer将数据分成了9:1，也可以用skf多折划分数据。
将标题 出处 描述三种文本分别加载roberta tokenizer，然后cat起来输入
数据量太少简单用了repeat增强，即将训练数据复制了一次。
'''


def create_dataloaders(args, test_mode=False):
    # 分割比率
    val_ratio = args.val_ratio
    anns = list()
    # 加载数据
    with open(args.train_annotation, 'r', encoding='utf8') as f:
        for line in f.readlines():
            ann = json.loads(line)
            anns.append(ann)
    # 洗牌and划分训练集，验证集
    random.shuffle(anns)
    val_anns = anns[:int(val_ratio * len(anns))]
    train_anns = anns[int(val_ratio * len(anns)):]
    # repeat <offline enhance>
    # 将训练集复制一份出来
    # train_anns = train_anns + train_anns
    # 建立数据集
    val_dataset = MultiModalDataset(args, val_anns)
    train_dataset = MultiModalDataset(args, train_anns)
    # 产生样例的策略的方式，是打乱随机还是顺序给
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    '''
    DataLoader()
    - dataset (Dataset) – 加载数据的数据集
    - batch_size (int, optional) – 每一轮加载多少样例，默认位1
    - sampler (Sampler or Iterable, optional) – 定义从数据集中产生样例的策略，可以在每一次迭代中由__len__实现。如果给定，则shuffle参数就不能给定了
    - pin_memory (bool, optional) – 如果该项位True，LoaderLoader将会在返回他们之前将TensorCopy到CUDA/Device的内存上，如果你的元素是自定义类型，
        或者是自定义类型的batch，请在自定义类上定义pin_memory()方法
    - num_workers (int, optional) – 多少线程用于加载数据. 0表示将会由main来加载. (default: 0)
    - prefetch_factor (int, optional, keyword-only arg) – 每次工作提前加载的batch数量. 2表示所有worker总共将预取 2 * num_workers 个批次。 
        (default: 2)
    '''
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    # 参数集，存数据的列表
    def __init__(self, args, anns, test_mode: bool = False):
        self.test_mode = test_mode
        # 加载tokenizer将字符串转为数值串
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.anns = anns

    def __len__(self) -> int:
        return len(self.anns)

    # idx:int(参数类型注释), ->dict(返回值类型注释)
    def __getitem__(self, idx: int) -> dict:
        id = self.anns[idx]['id']
        title = self.anns[idx]['title']
        assignee = self.anns[idx]['assignee']
        abstract = self.anns[idx]['abstract']
        # <online enhance here>
        # Step 2, load title tokens
        # text = title+assignee+abstract
        # text_inputs = self.tokenizer(title, max_length=512, padding='max_length', truncation=True)
        text_inputs = {}
        # tokenizer将各文本处理，返回dict:{'input_ids':[],'token_type_ids':[],'attention_mask':[]}
        title_inputs = self.tokenizer(title, max_length=30, padding='max_length', truncation=True)
        assignee_inputs = self.tokenizer(assignee, max_length=15, padding='max_length', truncation=True)
        abstract_inputs = self.tokenizer(abstract, max_length=450, padding='max_length', truncation=True)
        '''
        101 [CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。
        102 [SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。
        100 [UNK] 标志指的是未知字符
        103 [MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么
        '''
        # 一个用于bert学习的tensor应该以101开头，每两个句子用102隔开，最后结尾也以102结束
        title_inputs['input_ids'][0] = 101
        # 后面的词不需要前导101了
        assignee_inputs['input_ids'] = assignee_inputs['input_ids'][1:]
        abstract_inputs['input_ids'] = abstract_inputs['input_ids'][1:]
        assignee_inputs['attention_mask'] = assignee_inputs['attention_mask'][1:]
        abstract_inputs['attention_mask'] = abstract_inputs['attention_mask'][1:]
        assignee_inputs['token_type_ids'] = assignee_inputs['token_type_ids'][1:]
        abstract_inputs['token_type_ids'] = abstract_inputs['token_type_ids'][1:]
        for each in title_inputs:
            '''
            text_inputs['input_ids']=title_inputs['input_ids']+assignee_inputs['input_ids']+abstract_inputs['input_ids']
            ……
            '''
            text_inputs[each] = title_inputs[each] + assignee_inputs[each] + abstract_inputs[each]
        # 将dict中的所有item转为tensor
        text_inputs = {k: torch.LongTensor(v) for k, v in text_inputs.items()}
        data = dict(
            text_inputs=text_inputs['input_ids'],
            text_mask=text_inputs['attention_mask'],
            text_type_ids=text_inputs['token_type_ids'],
        )
        # Step 4, load label if not test mode
        if not self.test_mode:
            data['label'] = torch.LongTensor([self.anns[idx]['label_id']])
        # 返回的数据是一个dict类型
        return data
