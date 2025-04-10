import json
from transformers import BertTokenizer, BertConfig, LlamaTokenizer, LlamaForCausalLM
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import LJPDataSet
from LawKeywordBert.models.bert_for_ner import BertSpanForNer
import pandas as pd
from typing import List, Dict
from utils import get_metrics, labelmap


def get_parse():
    parser = argparse.ArgumentParser('KnowJudge')
    parser.add_argument('--model_path', type=str, default="models/lawyerllama",
                        help='The model file path.')
    parser.add_argument('--model_type', type=str, default="llama",
                        help='The model type.')
    parser.add_argument('--bert_model_path', type=str, default="LawKeywordBert/outputs",
                        help='The bert model file path.')
    parser.add_argument('--data_path', type=str, default='data/cail2018.json',
                        help='The learning rate.')
    parser.add_argument('--charge_list', type=list,
                        default=['危险驾驶', '盗窃', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品'],
                        help='The charge list of classify.')
    parser.add_argument('--precedent_path', type=str, default='data/precedent_database/precedent_data.json',
                        help='The precedent file path.')
    parser.add_argument('--precedent_pt_path', type=str, default='data/precedent_database/keyword_precedent_db.pt',
                        help='The precedent file path.')
    parser.add_argument('--crime_law_path', type=str, default='data/other_data/crime_law.csv',
                        help='The crime law file path.')
    parser.add_argument('--llm_max_input_length', type=int, default=1024,
                        help='The maximum input sequence length of llm.')
    parser.add_argument("--bert_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument('--k_keywords', type=int, default=20)
    parser.add_argument('--n_keywords', type=int, default=10)
    parser.add_argument('--bert_ner_task_name', type=str, default='keyword',
                        help='The task name of bert ner.')
    parser.add_argument('--keyword_method', type=str, default='ner',
                        help='The method of LKR extraction.')
    parser.add_argument('--diverse_method', type=str, default='gnn',
                        help='The method of LKR diverse process in LKR extraction.')
    parser.add_argument('--use_keywords', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Using keywords for article retrieval.')
    parser.add_argument('--max_output_length', type=int, default=100,
                        help='The max output length.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='The batch size.')

    return parser


def load_model(
        model_path: str = 'chinese_alpaca_2_1.3b_rlhf',
        model_type: str = 'llama',
):
    print('{} model is used.'.format(model_type))
    if model_type == 'llama':
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            do_sample=False,
        ).to(device)

    else:
        raise ValueError('The model type {} is not supported.'.format(model_type))

    return model


def load_tokenizer(
        tokenizer_path: str = 'chinese_alpaca_2_1.3b_rlhf',
        model_type: str = 'llama',
):
    if model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, padding_side='left')
    else:
        raise ValueError('The {} tokenizer type is not supported.'.format(model_type))

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def load_data(
        batch_size: int = 2,
        data_path: str = '',
        charge_list: list = None,
        precedent_db: List[Dict] = None,
        precedent_pt: torch.Tensor = None,
        crime_law: pd.DataFrame = None,
        k: int = 20,
        n: int = 10,
        task_name: str = 'LKR',
        bert_max_seq_length: int = 512,
        llm_tokenizer: LlamaTokenizer = None,
        max_output_length: int = 10,
        keyword_method: str = 'ner',
        diverse_method: str = 'gnn',
        use_keywords: bool = True,
):
    if charge_list is None:
        charge_list = ['危险驾驶', '盗窃', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品']

    dataset = LJPDataSet(
        data_path=data_path,
        charge_list=charge_list,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        bert_config=bert_config,
        precedent_db=precedent_db,
        precedent_pt=precedent_pt,
        crime_law=crime_law,
        k=k,
        n=n,
        task_name=task_name,
        bert_max_seq_length=bert_max_seq_length,
        llm_tokenizer=llm_tokenizer,
        max_output_length=max_output_length,
        keyword_method=keyword_method,
        diverse_method=diverse_method,
        use_keywords=use_keywords,
    )

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataloader


def inference(
        model,
        precedent_db: List[Dict] = None,
        precedent_pt: torch.Tensor = None,
        crime_law: pd.DataFrame = None,
        max_output_length: int = 13,
        result_path: str = '',
):
    val_tokenizer = load_tokenizer(args.model_path, model_type=args.model_type)

    val_data = load_data(
        batch_size=args.batch_size,
        data_path=args.data_path,
        charge_list=args.charge_list,
        precedent_db=precedent_db,
        precedent_pt=precedent_pt,
        crime_law=crime_law,
        k=args.k_keywords,
        n=args.n_keywords,
        task_name=args.bert_ner_task_name,
        bert_max_seq_length=args.bert_max_seq_length,
        llm_tokenizer=val_tokenizer,
        max_output_length=args.max_output_length,
        keyword_method=args.keyword_method,
        diverse_method=args.diverse_method,
        use_keywords=args.use_keywords,
    )

    model.to(device)

    model.eval()

    total_val_data = len(val_data)

    print('{} data will be inferred. Size is {}.'.format(total_val_data * args.val_batch_size,
                                                         total_val_data))

    y_pred = []
    y_true = []

    res = pd.DataFrame(columns=['fact', 'charge', 'prediction'])

    for i, sample in enumerate(val_data):
        if (i + 1) % 100 == 0:
            print('{}/{}'.format(i + 1, total_val_data))

        # batch size > 1
        inputs = [input_text for input_text in sample["inputs"]]

        input_ids = val_tokenizer(inputs,
                                  return_tensors="pt",
                                  padding='max_length',
                                  max_length=args.llm_max_input_length,
                                  truncation=True).input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=max_output_length, do_sample=False, repetition_penalty=1.1)
        output_texts = val_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_id = []
        for output_text in output_texts:
            output_text = output_text.replace('：', ':')
            output_text = output_text.replace(' ', '')
            output_text = output_text.split('问题:被告将以什么罪进行判决')[-1].strip()
            # print(output_text)
            pred_id.append(labelmap(output_text, args.charge_list))
        y_pred.extend(pred_id)
        y_true.extend(sample["charge_id"].detach().cpu().numpy().tolist())

    print('*' * 10, 'Classification Report', '*' * 10)
    get_metrics(y_true, y_pred, args.charge_list)
    print('*' * 50)

    if result_path:
        res.to_csv(result_path, index=False)


def prepare_precedent():
    # prepare precedent_data
    with open(args.precedent_path, 'r', encoding='utf-8') as f:
        precedent_db = json.load(f)
    precedent_pt = torch.load(args.precedent_pt_path).to(device)

    # prepare crime_law
    crime_law = pd.read_csv(args.crime_law_path)

    return precedent_db, precedent_pt, crime_law


if __name__ == '__main__':
    args = get_parse().parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    bert_config = BertConfig.from_pretrained(args.bert_model_path)
    bert_model = BertSpanForNer.from_pretrained(args.bert_model_path, config=bert_config).to(device)

    precedent_db, precedent_pt, crime_law = prepare_precedent()

    model = load_model(args.model_path, model_type=args.model_type).to(device)

    print('*' * 30, 'Test Metrics', '*' * 30)
    inference(
        model,
        max_output_length=args.max_output_length,
        precedent_db=precedent_db,
        precedent_pt=precedent_pt,
        crime_law=crime_law,
        # result_path='',
    )
