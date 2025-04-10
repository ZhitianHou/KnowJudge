import torch
from bert_ner.models.bert_for_ner import BertSpanForNer
from transformers import BertTokenizer, BertConfig, AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F


def get_cls_token(
        inputs: str,
        tokenizer: BertTokenizer,
        model: BertSpanForNer,
        config: BertConfig,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        for_word: bool = False,
) -> torch.Tensor:
    model.eval()

    cls = torch.zeros(1, config.hidden_size).to(device)

    if for_word:
        model_inputs = tokenizer(inputs[:510], return_tensors="pt").to(device)
        _ = model(**model_inputs)

    else:
        while len(inputs) >= 510:
            # search the punctuation for segment
            segment_pun = ['。', '；', ';', '，', ',']
            period_index = -1
            for pun in segment_pun:
                period_index = inputs[:510].rfind(pun)
                if period_index != -1:
                    break

            model_inputs = tokenizer(inputs[:period_index+1], return_tensors="pt").to(device)
            _ = model(**model_inputs)

            cls += model.last_hidden_state[0, 0:1, :]
            inputs = inputs[period_index + 1:]

        if len(inputs) >= 10:
            model_inputs = tokenizer(inputs[:510], return_tensors="pt").to(device)
            _ = model(**model_inputs)

    # print(model.last_hidden_state.shape)
    cls += model.last_hidden_state[0, 0:1, :]
    cls = cls / torch.norm(cls)

    return cls


def add_keyword2cls(
        tokenizer,
        model,
        config,
        pre_fact_cls,
        keywords: list,
):
    if len(keywords) == 1 and keywords[0] == '无':
        pass
    else:
        for keyword in keywords:
            pre_fact_cls += get_cls_token(keyword, tokenizer, model, config, for_word=True)
        pre_fact_cls /= len(keywords) + 1
    return pre_fact_cls


def labelmap(
        model_output: str,
        charge_list: list = None,
        model_path: str = 'models/simcse-bert-base-uncased',
):
    if charge_list is None:
        charge_list = ['危险驾驶', '盗窃', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品']

    for i, charge in enumerate(charge_list):
        if charge in model_output:
            return i

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    texts = ['被告将以' + charge + '罪进行判决' for charge in charge_list]
    texts.append(model_output)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    return torch.argmax(F.cosine_similarity(embeddings[:-1], embeddings[-1:], dim=-1)).item()


def get_metrics(
        y_true: list,
        y_pred: list,
        target_names: list = None,
):
    if target_names is None:
        target_names = ['危险驾驶', '盗窃', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品']

    print(classification_report(y_true, y_pred, target_names=target_names, labels=range(len(target_names))))
    print(accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    pass
