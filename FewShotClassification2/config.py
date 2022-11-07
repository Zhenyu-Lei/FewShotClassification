import argparse

data_path = './data/'


def parse_args():
    parser = argparse.ArgumentParser(description="Few Shot Classification For MachineLearning course")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--cls_dropout', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--ema', type=bool, default=True, help='ema')
    parser.add_argument('--attack', type=str, default=None, help='attack')
    parser.add_argument('--use_fp16', type=bool, default=False, help='fp16')
    parser.add_argument('--all', type=bool, default=False, help='all_data')
    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default=data_path + 'data_enhance.json')
    parser.add_argument('--val_annotation', type=str, default=data_path + 'val_div.json')
    parser.add_argument('--test_annotation', type=str, default=data_path + 'testA.json')
    parser.add_argument('--test_output_csv', type=str, default=data_path + 'submission.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float,
                        help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=128, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=512, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save')
    parser.add_argument('--ckpt_file', type=str, default='/media/zsy/CCF/save/flod_/model_epoch_19_mean_f1_0.6616.bin')
    parser.add_argument('--best_score', default=-0.5, type=float, help='save checkpoint if mean_f1 > best_score')
    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=200, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='./bert')
    parser.add_argument('--test_bert_dir', type=str, default='roberta_wwm_chinese')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--bert_output_dim", type=float, default=768)
    parser.add_argument("--bert_hidden_size", type=float, default=768)
    parser.add_argument("--bert_class_num", type=int, default=36)
    return parser.parse_args()
