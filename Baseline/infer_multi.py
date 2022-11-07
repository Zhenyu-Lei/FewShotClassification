import json

import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from config import parse_args
from data_helper import MultiModalDataset
from model import clsModel
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def inference():
    args = parse_args()
    # 1. load data
    anns = list()
    with open(args.test_annotation, 'r', encoding='utf8') as f:
        for line in f.readlines():
            ann = json.loads(line)
            anns.append(ann)
    dataset = MultiModalDataset(args, anns)
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
    for i in range(5):
        model = clsModel(args)
        save_path = f'save/flod_{i}'
        best_model = os.path.join(save_path, 'model_best.bin')
        checkpoint = torch.load(best_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        models.append(model)
    # 3. inference
    all_outs = []
    for model in models:
        print('infering')
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
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
