import json

import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from transformers import BertForSequenceClassification

from config import parse_args
from data_helper import MultiModalDataset
from model import clsModel
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

model_description = {
    0: [0, "model_epoch_9_mean_f1_0.5813.bin"],
    1: [0, "2E-4_epoch5_0.5255.bin"],
    2: [1, "model_epoch_13_mean_f1_0.6449.bin"],
    3: [1, "model_epoch_19_mean_f1_0.6232.bin"],
    4: [1, "model_epoch_5_mean_f1_0.5038.bin"]
}


def inference():
    args = parse_args()
    # 1. load data
    anns = list()
    with open(args.test_annotation, 'r', encoding='utf8') as f:
        for line in f.readlines():
            ann = json.loads(line)
            anns.append(ann)
    dataset = MultiModalDataset(args, anns,True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    # 2. load model i
    models = []
    for idx, description in model_description.items():

        save_path = f'save/flod_{idx}'
        model_dict = os.path.join(save_path, description[1])
        if description[0] == 0:
            model = BertForSequenceClassification.from_pretrained(args.bert_dir, num_labels=args.bert_class_num)
        else:
            model = clsModel(args)
        checkpoint = torch.load(model_dict, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        models.append([description[0], model])

    # 3. inference
    all_outs = []
    for model_pair in models:
        model = model_pair[1]
        print('infering')
        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if model_pair[0] == 0:
                    output = model(batch['text_inputs'], batch['text_mask'],
                                   labels=torch.zeros([len(batch['text_inputs']), 36]))
                    outs = output[1]
                else:
                    outs = model(batch, inference=True, multi=True)
                predictions.extend(outs.cpu().numpy())
            predictions = np.array(predictions)
            all_outs.append(predictions)
    all_outs = np.array(all_outs)
    out = np.sum(all_outs, axis=0)
    predictions = np.argmax(out, axis=1)
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        f.write(f'id,label\n')
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            f.write(f'{video_id},{pred_label_id}\n')


if __name__ == '__main__':
    inference()
