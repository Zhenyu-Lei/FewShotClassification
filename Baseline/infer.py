import json

import torch
from torch.utils.data import SequentialSampler, DataLoader
import os
from config import parse_args
from model import clsModel
from tqdm import tqdm
from data_helper import MultiModalDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def inference():
    args = parse_args()
    print(args.ckpt_file)
    print(args.test_batch_size)
    anns = list()
    with open(args.test_annotation, 'r', encoding='utf8') as f:
        for line in f.readlines():
            ann = json.loads(line)
            anns.append(ann)
    dataset = MultiModalDataset(args, anns, True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    # 2. load model
    model = clsModel(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    new_key = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # model.half()
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pred_label_id = model(data=batch, inference=True)
            predictions.extend(pred_label_id)
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        f.write(f'id,label\n')
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            f.write(f'{video_id},{pred_label_id}\n')


if __name__ == '__main__':
    inference()
