import os
import random
import logging
import argparse

import torch
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

from models.bert_for_ner import BertCrfForNer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model_service.pytorch_model_service import PTServingBaseService
except:
    PTServingBaseService = object

    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="cner", type=str,
                        help="The name of the task to train selected in the list: ")
    parser.add_argument("--data_dir", default="./datasets/cner/", type=str,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default="./outputs/cner_output/bert/", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " )
    parser.add_argument("--output_dir", default="./outputs/predict_output/bert/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    # Other parameters
    parser.add_argument('--markup', default='bios', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    # adversarial training
    parser.add_argument("--do_adv", action="store_true",
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints",type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    return parser
    
def read_data(filepath):
    sentences = []
    sent = ['[START]']
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            if line == '\n':
                if len(sent) > 1:
                    sentences.append(sent + ['[END]'])
                sent = ['[START]']
            else:
                sent.append(line[0])
    return sentences


def batch_iter(data, batch_size=24, shuffle=True):
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x), reverse=True)
        yield batch
    
    
class CustomizeService(PTServingBaseService):
    def __init__(self, model_name, model_path):  # model_name, model_path 没用
        self.args = get_args().parse_args()
        MODEL_CLASSES = {
            'bert': (BertConfig, BertCrfForNer, BertTokenizer),
        }
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES["bert"]

    def _preprocess(self, data):
        preprocessed_data = {}
        for _, v in data.items():
            for file_name, file_content in v.items():
                with open(file_name, "wb") as f:
                    f.write(file_content.read())
                sentences = read_data(file_name)
                # sentences = utils.words2indices(sentences, self.sent_vocab)
                preprocessed_data[file_name] = sentences
        # print(preprocessed_data)
        return preprocessed_data

    def _inference(self, data):
        config = self.config_class.from_pretrained(self.args.model_name_or_path,num_labels=23,)
        tokenizer = self.tokenizer_class.from_pretrained(self.args.model_name_or_path, do_lower_case=self.args.do_lower_case)

        checkpoints = [self.args.output_dir]
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(self.args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(self.args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(self.args.device)
            predict(self.args, model, tokenizer, prefix=prefix)

    def _postprocess(self, data) -> dict:
        logger.info("in postprocess")
        return data