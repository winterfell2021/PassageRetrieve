import torch
import wandb
from Utils.LoggerUtil import LoggerUtil
from typing import Dict
from Config.TrainConfig import TrainConfig
import logging
import os

def get_logger(log_name="Vector"):
    logger = logging.getLogger(log_name)
    return logger


def init_logger(log_name, log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(filename)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = logging.FileHandler(log_file, encoding="utf8", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)



class VectorInfoLogger():
    def __init__(self, conf: TrainConfig):
        self.conf = conf
        self.best_eval_result = None
        self.wandb = conf.wandb
        self.logger = get_logger()

    def try_print_log(self, loss: torch.Tensor, eval_result: Dict, step: int, global_step: int, epoch_steps: int,
                      epoch: int, num_epochs: int, *args, **kwargs):
        # 尝试输出loss
        if step % self.conf.log_step == 0:
            print_string = "epoch-{},\tstep:{}/{},\tloss:{}".format(
                epoch, step, epoch_steps, loss.data)
        else:
            print_string = ""
        if len(print_string) > 5:
            if self.wandb:
                wandb.log({"loss": loss.data, "epoch": epoch,
                          "step": step}, step=global_step)
            self.logger.info(print_string)
        if eval_result is not None:
            # 尝试输出本次评测结果以及最优结果
            self.logger.info("=" * 15 + "本次评测的最终信息" + "=" * 15)
            print_string = ""
            print_string += "本次测评结果是："
            for k, v in eval_result.items():
                print_string += "{}:{},\t".format(k, round(v, 5))
            if self.wandb:
                wandb.log(eval_result, step=global_step)
            self.logger.info(print_string)
            # 判断当前指标是否最好
            if self.best_eval_result is not None:
                for metric in self.conf.eval_metrics:
                    if eval_result[metric] > self.best_eval_result[metric]:
                        self.logger.info("获取到了更高的指标:{}".format(metric))
                        self.best_eval_result[metric] = eval_result[metric]
                eval_str = "当前最优指标是："
                for k, v in self.best_eval_result.items():
                    eval_str += "{}:{},\t".format(k, round(v, 5))
                    if self.wandb:
                        wandb.run.summary[k] = round(v, 5)
                self.logger.info(eval_str)
            else:
                self.best_eval_result = eval_result

        # 输出相关输入信息 主要用于调试
        if self.conf.print_input_step > 0 and global_step % self.conf.print_input_step == 0:
            self.logger.info("=" * 30 + "随机输出一组输入信息" + "=" * 30)
            ipt = kwargs["ipt"]
            tokenizer = kwargs["tokenizer"]

            ##########################################################################################
            ipt_ids = ipt["query_ipt"]["input_ids"][0].cpu().numpy().tolist()
            token_type_ids = ipt["query_ipt"]["token_type_ids"][0].cpu(
            ).numpy().tolist()
            attn_mask = ipt["query_ipt"]["attention_mask"][0].cpu(
            ).numpy().tolist()
            ipt_tokens = tokenizer.convert_ids_to_tokens(ipt_ids)
            self.logger.info("ipt_ids:{}".format(
                ",".join([str(i) for i in ipt_ids])))
            self.logger.info("ipt_tokens:{}".format(",".join(ipt_tokens)))
            self.logger.info("token_type_ids:{}".format(
                ",".join([str(i) for i in token_type_ids])))
            self.logger.info("attn_mask:{}".format(
                ",".join([str(i) for i in attn_mask])))
            ##########################################################################################
            ipt_ids = ipt["doc_ipt"]["input_ids"][0].cpu().numpy().tolist()
            token_type_ids = ipt["doc_ipt"]["token_type_ids"][0].cpu(
            ).numpy().tolist()
            attn_mask = ipt["doc_ipt"]["attention_mask"][0].cpu(
            ).numpy().tolist()
            ipt_tokens = tokenizer.convert_ids_to_tokens(ipt_ids)
            self.logger.info("ipt_ids:{}".format(
                ",".join([str(i) for i in ipt_ids])))
            self.logger.info("ipt_tokens:{}".format(",".join(ipt_tokens)))
            self.logger.info("token_type_ids:{}".format(
                ",".join([str(i) for i in token_type_ids])))
            self.logger.info("attn_mask:{}".format(
                ",".join([str(i) for i in attn_mask])))
