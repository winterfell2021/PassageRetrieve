import argparse
# from simpletransformers.language_modeling import LanguageModelingModel
from datetime import datetime
from model import LanguageModelingModel
from utils import init_logger, get_logger
import os
if __name__ == '__main__':
    train_args = {
        "tokenizer_name": "./config",
        "config_name": "./config",
        "reprocess_input_data": False,
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "save_eval_checkpoints": True,
        "block_size": 509,
        "max_seq_length": 509,
        "save_model_every_epoch": True,
        "learning_rate": 1e-4,
        "train_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "mlm": True,
        "dataset_type": "simple",
        "logging_steps": 100,
        "evaluate_during_training": False,
        "sliding_window": True,
        "use_multiprocessing": False,
        "vocab_size": 30522,
        "fp16": False,
        "local_rank": -1,
        "wandb_project": "DFPassageRetrieve",
        "handle_chinese_chars": False
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='./data/full.txt', help='data directory')
    parser.add_argument('--output', type=str, dest="output_dir",
                        default='./output/', help='work dir')
    parser.add_argument('--model', type=str,
                        default='roformer_v2', help='model name')
    parser.add_argument('--config', dest="config_name", type=str, default="./config/roformer_v2_large/")
    parser.add_argument('-b', '--batch-size', dest="train_batch_size",
                        type=int, default=16, help='batch size')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--resume", dest="model_name", type=str, default=None,
                        help="")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.model)
    if os.path.exists(args.output_dir):
        args.output_dir += '_' + datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)
    init_logger("pretrain", os.path.join(args.output_dir, "pretrain.log"))
    logger = get_logger("pretrain")
    logger.info(args)
    
    for key, value in vars(args).items():
        train_args[key] = value
    train_args['tokenizer_name'] = train_args['config_name']
    train_args['config_name'] = train_args['config_name']
    train_args['best_model_dir'] = os.path.join(train_args['output_dir'], 'best_model')
    model = LanguageModelingModel(
        args.model,
        args.model_name,
        args=train_args
    )
    model.train_model(
        train_file=args.input,
    )
