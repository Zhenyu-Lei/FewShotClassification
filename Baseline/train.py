import logging
import os
import time
import torch
from config import parse_args
from data_helper import create_dataloaders
from model import clsModel
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
from util import *
import json
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import autocast as ac


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    model.train()
    return loss, results


def train_and_validate(args):
    # 创建存储模型文件的位置
    if not os.path.exists(f'{args.savedmodel_path}/flod_'): os.makedirs(f'{args.savedmodel_path}/flod_')
    # 加载数据
    train_dataloader, val_dataloader = create_dataloaders(args)
    # 配置模型
    model = clsModel(args)
    # 尝试冻结
    # unfreeze_layers = ['layer.10','layer.11','bert.pooler','out.']
    # for name ,param in model.bert.named_parameters():
    #     param.requires_grad = False
    #     for ele in unfreeze_layers:
    #         if ele in name:
    #             param.requires_grad = True
    #             break

    # 配置优化器and学习率变化器
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    # -------ema here-----------------
    # （Exponential Moving Average）指数移动平均值，需要提供一个衰减率（decay）。这个衰减率将用于控制模型的更新速度。
    if args.ema:
        ema = EMA(model, 0.999)
        ema.register()
    fgm, pgd = None, None
    '''
    对抗样本：
    通常是对原始的输入添加一定的扰动来构造，然后放给模型训练，这样模型就有了识别对抗样本的能力。
    其中的关键技术在于如果构造扰动，使得模型在不同的攻击样本中均能够具备较强的识别性。
    '''
    if args.attack == 'fgm':
        fgm = FGM(model=model)
        print('fgming')
    elif args.attack == 'pgd':
        pgd = PGD(model=model)
        pgd_k = 3
        print('pgding')
    # 是否使用FP16数据类型
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    model.train()
    loss, results = validate(model, val_dataloader)
    # -------------------------------
    # 3. training
    step = 0
    # save checkpoint if mean_f1 > best_score
    best_score = args.best_score
    start_time = time.time()
    # 总步数为train_dataloader的长度和epoch的个数
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for i, batch in enumerate(train_dataloader):
            # 使用bn层以及dropout
            model.train()
            if args.use_fp16:
                # 半精度训练
                with ac():
                    loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
            # 使用生成对抗数据
            if fgm is not None:
                fgm.attack()
                if args.use_fp16:
                    with ac():
                        loss_adv, _, _, _ = model(batch)
                else:
                    loss_adv, _, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                if args.use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()
                fgm.restore()
            elif pgd is not None:
                pgd.backup_grad()
                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))
                    if _t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    if args.use_fp16:
                        with ac():
                            loss_adv, _, _, _ = model(batch)
                    else:
                        loss_adv, _, _, _ = model(batch)
                    loss_adv = loss_adv.mean()
                    if args.use_fp16:
                        scaler.scale(loss_adv).backward()
                    else:
                        loss_adv.backward()
                pgd.restore()
            if args.use_fp16:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            model.zero_grad()
            scheduler.step()
            if args.ema:
                # ------ema update--------
                ema.update()
                # ------------------------
            step += 1
            if i % (100000 // args.batch_size // 4) == 0 and 0 < i < (
                    100000 // args.batch_size - 100000 // args.batch_size // 3 - 100) and epoch > 1:
                if args.ema:
                    # --------ema shadow--------
                    ema.apply_shadow()
                    # --------------------------
                loss, results = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                mean_f1 = results['mean_f1']
                if mean_f1 >= best_score:
                    best_score = mean_f1
                    torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/flod_/model_epoch_{epoch}_{i}_mean_f1_{mean_f1}.bin')
                    best_score = mean_f1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(
                    f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
        if args.ema:
            # --------ema shadow--------
            ema.apply_shadow()
            # --------------------------
        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        # 5. save checkpoint
        mean_f1 = results['f1_macro']
        if mean_f1 > best_score:
            best_score = mean_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/flod_/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        if args.ema:
            # --------ema restore-------
            ema.restore()
            # --------------------------


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)


if __name__ == '__main__':
    main()
