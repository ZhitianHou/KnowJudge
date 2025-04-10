import pandas as pd
import torch
from transformers import BertTokenizer, BertConfig
from LawKeywordBert.models.bert_for_ner import BertSpanForNer
from keyword_extraction import keyword_prediction
from law_retrival import reference_law_by_precedent
from typing import List, Dict


def generate_template(
        case: str,
        model: BertSpanForNer,
        tokenizer: BertTokenizer,
        config: BertConfig,
        precedent_db: List[Dict],
        precedent_pt: torch.Tensor,
        crime_law: pd.DataFrame,
        k: int = 10,
        n: int = 10,
        task_name: str = 'LKR',
        max_seq_length: int = 512,
        device: torch.device = torch.device('cpu'),
        keyword_method: str = 'ner',
        diverse_method: str = 'gnn',
        use_keywords: bool = True,
):
    case = case.replace('：', ':')
    case = case.replace(':', '，')
    keywords = keyword_prediction(tokenizer, model, config, task_name, case, k=k, n=n,
                                  max_seq_length=max_seq_length,
                                  device=device,
                                  method=keyword_method,
                                  diverse_method=diverse_method)
    # keywords = ['无']
    # articles = '无'
    articles = reference_law_by_precedent(
        precedent_db,
        case,
        precedent_pt,
        crime_law,
        tokenizer,
        model,
        config,
        use_keywords=use_keywords,
        keywords=keywords,
    )
    keywords = '，'.join(keywords)
    template = f'事实描述：{case}\n关键词：{keywords}\n参考法条：{articles}'

    return template


if __name__ == '__main__':
    pass
