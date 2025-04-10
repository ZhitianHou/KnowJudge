from typing import List, Dict
import pandas as pd
from torch.utils.data import Dataset
import json
from bert_ner.models.bert_for_ner import BertSpanForNer
from transformers import BertTokenizer, BertConfig, LlamaTokenizer
from generate_template import generate_template
import torch


class LJPDataSet(Dataset):
    def __init__(
            self,
            data_path: str = '',
            charge_list: list = None,
            bert_model: BertSpanForNer = None,
            bert_tokenizer: BertTokenizer = None,
            bert_config: BertConfig = None,
            precedent_db: List[Dict] = None,
            precedent_pt: torch.Tensor = None,
            crime_law: pd.DataFrame = None,
            k: int = 20,
            n: int = 10,
            task_name: str = 'keyword',
            bert_max_seq_length: int = 512,
            llm_tokenizer: LlamaTokenizer = None,
            max_output_length: int = 10,
            num_examples: int = 2000,
            keyword_method: str = 'ner',
            diverse_method: str = 'gnn',
            use_keywords: bool = True,
    ):
        self.data = None
        if charge_list is None:
            charge_list = ['危险驾驶', '盗窃', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品']
        self.charge2idx = {charge: i for i, charge in enumerate(charge_list)}
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_config = bert_config
        self.precedent_db = precedent_db
        self.precedent_pt = precedent_pt
        self.crime_law = crime_law
        self.k = k
        self.n = n
        self.task_name = task_name
        self.bert_max_seq_length = bert_max_seq_length
        self.llm_tokenizer = llm_tokenizer
        self.max_output_length = max_output_length
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_examples = num_examples
        self.data_path = data_path
        self.keyword_method = keyword_method
        self.diverse_method = diverse_method
        self.use_keywords = use_keywords
        self.load_data()

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[:self.num_examples]

    def transform_text(self, text):
        text = generate_template(
            text,
            self.bert_model,
            self.bert_tokenizer,
            self.bert_config,
            self.precedent_db,
            self.precedent_pt,
            self.crime_law,
            self.k,
            self.n,
            self.task_name,
            max_seq_length=self.bert_max_seq_length,
            device=self.device,
            keyword_method=self.keyword_method,
            diverse_method=self.diverse_method,
            use_keywords=self.use_keywords,
        )
        text = "### Human: " + text + "\n问题：被告将以什么罪进行判决\n### Assistant: "
        return text

    def transform_label(self, charge: str):
        return torch.tensor(self.charge2idx[charge]).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "inputs": self.transform_text(self.data[item]['case']['fact']),
            "charge": self.data[item]['case']['charge'],
            "charge_id": self.transform_label(self.data[item]['case']['charge'])
        }
