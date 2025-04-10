import cn2an
import pandas as pd
import torch
from utils import get_cls_token, add_keyword2cls
from transformers import BertTokenizer, BertConfig
from LawKeywordBert.models.bert_for_ner import BertSpanForNer
import torch.nn.functional as F
from typing import List, Dict


def reference_law_by_precedent(
        precedent_db: List[Dict],
        case: str,
        precedent_pt: torch.Tensor,
        crime_law: pd.DataFrame,
        tokenizer: BertTokenizer,
        model: BertSpanForNer,
        config: BertConfig,
        use_keywords: bool = True,
        keywords: list = None,
):
    assert not (use_keywords and keywords is None), "When use_keywords is True, keywords cannot be None"

    case_cls = get_cls_token(case, tokenizer=tokenizer, model=model, config=config)

    if use_keywords:
        case_cls = add_keyword2cls(tokenizer, model, config, case_cls, keywords)

    scores = F.cosine_similarity(precedent_pt, case_cls, dim=-1)

    max_similarity_index = torch.argmax(scores).item()

    # print(precedent_db[max_similarity_index]['case']['fact'])

    charge2article = {
        '危险驾驶': 133,
        '盗窃': 264,
        '故意伤害': 234,
        '交通肇事': 133,
        '走私、贩卖、运输、制造毒品': 347,
    }
    article = charge2article[precedent_db[max_similarity_index]['case']['charge']]
    article_text = crime_law[crime_law['article'] == article]['law'].values.tolist()[0]

    if precedent_db[max_similarity_index]['case']['charge'] == '危险驾驶':
        article_text = article_text.split('第一百三十三条之一')[-1]
        reference_law = '第' + cn2an.an2cn(article) + '条之一：' + article_text
    elif precedent_db[max_similarity_index]['case']['charge'] == '交通肇事':
        article_text = article_text.split('第一百三十三条之一')[0]
        reference_law = '第' + cn2an.an2cn(article) + '条：' + article_text
    else:
        reference_law = '第' + cn2an.an2cn(article) + '条：' + article_text
    return reference_law


if __name__ == '__main__':
    pass
