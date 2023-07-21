import os
import logging
import argparse
import glob
import json

import torch
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

from models.bert_for_ner import BertCrfForNer
from run_ner_crf import predict
from processors.ner_seq import CnerProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model_service.pytorch_model_service import PTServingBaseService
except:
    PTServingBaseService = object

MODEL_NUM = 1232


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


class CustomizeService(PTServingBaseService):

    def save_data(self, data):
        dataset = list(data.values())[0]
        out_data = []
        for line in dataset:
            for c in line:
                if c != "[START]" and c != "[END]":
                    out_data.append(c + " O")
            out_data.append("\n")

        with open(os.path.join(self.code_url, "datasets/cner/test.char.bmes"), "w", encoding="utf-8") as f:
            f.write("N O")
            f.write("\n")
            f.write("U O")
            f.write("\n")
            f.write("L O")
            f.write("\n")
            f.write("L O")
            f.write("\n")
            f.write("\n")

            for i in out_data:
                f.write(i)
                if i != "\n":
                    f.write("\n")

    def get_res(self):
        res = {"result": []}
        words = []
        line = []
        with open(os.path.join(self.code_url, "datasets/cner/test.char.bmes"), "r", encoding="utf-8") as f:
            for i in f.readlines():
                if i != "\n":
                    line.append(i[0])
                else:
                    words.append(line)
                    line = []
        words = words[1:]

        tags = []
        with open(os.path.join(self.code_url, f"outputs/cner_output/bert/checkpoint-{MODEL_NUM}/test_prediction.json"), "r", encoding="utf-8") as f:
            for i in f.readlines():
                d = json.loads(i)["tag_seq"]
                tags.append(d.split(" "))

        print("-" * 20)
        print(f"len(words): {len(words)}\nlen(tags): {len(tags)}")
        print("-" * 20)
        assert len(words) == len(tags)

        for i in range(len(tags)):
            for j in range(len(words[i])):
                tag = tags[i][j]
                if tags[i][j] == "B-TITLE":
                    tag = "B-DATE"
                elif tags[i][j] == "I-TITLE":
                    tag = "I-DATE"
                elif tags[i][j] == "B-NAME":
                    tag = "B-PER"
                elif tags[i][j] == "I-NAME":
                    tag = "I-PER"

                res["result"].append(f"{words[i][j]} {tag}\n")
            res["result"].append("\n")

        # with open(os.path.join(self.code_url, "ress.txt"), "w", encoding="utf-8") as f:
        #     f.writelines(res.get("result"))

        # print(res)
        return res

    def __init__(self, model_name, model_path):  # model_name, model_path 没用
        self.code_url = os.path.dirname(os.path.abspath(__file__))
        self.args = argparse.Namespace()
        print(type(self.args))
        MODEL_CLASSES = {
            'bert': (BertConfig, BertCrfForNer, BertTokenizer),
        }
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES["bert"]

        # init args
        self.args.data_dir = os.path.join(self.code_url, "datasets/cner/")
        self.args.model_name_or_path = os.path.join(self.code_url, "prev_trained_model/bert-base-chinese")
        self.args.output_dir = os.path.join(self.code_url, "outputs/cner_output/bert/")
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.do_eval = False
        self.args.do_train = False
        self.args.do_predict = True
        self.args.do_lower_case = False
        self.args.predict_checkpoints = MODEL_NUM
        self.args.task_name = "cner"
        self.args.model_type = "bert"
        self.args.markup = "bios"
        self.args.local_rank = -1
        self.args.overwrite_cache = False
        self.args.eval_max_seq_length = 512
        self.args.train_max_seq_length = 128
        self.args.gradient_accumulation_steps = 1
        self.args.logging_steps = 50
        self.args.warmup_proportion = 0.1
        self.args.seed = 42
        self.args.fp16_opt_level = "O1"
        self.args.max_steps = -1
        processor = CnerProcessor()
        label_list = processor.get_labels()
        self.args.id2label = {i: label for i, label in enumerate(label_list)}
        self.args.label2id = {label: i for i, label in enumerate(label_list)}

    def _preprocess(self, data):
        preprocessed_data = {}
        for _, v in data.items():
            for file_name, file_content in v.items():
                with open(file_name, "wb") as f:
                    f.write(file_content.read())
                sentences = read_data(file_name)
                preprocessed_data[file_name] = sentences
        print(preprocessed_data)
        self.save_data(preprocessed_data)
        return preprocessed_data

    def _inference(self, data):
        config = self.config_class.from_pretrained(self.args.model_name_or_path, num_labels=23, )
        tokenizer = self.tokenizer_class.from_pretrained(self.args.model_name_or_path,
                                                         do_lower_case=self.args.do_lower_case)

        checkpoints = [self.args.output_dir]
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(self.args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(self.args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = self.model_class.from_pretrained(checkpoint, config=config)
            model.to(self.args.device)
            predict(self.args, model, tokenizer, prefix=prefix)
        return data

    def _postprocess(self, data) -> dict:
        # 输出格式是{"result":["x O\n","x B-DATE\n"]}
        data = self.get_res()
        return data
