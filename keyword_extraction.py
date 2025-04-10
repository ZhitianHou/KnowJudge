# -- coding:utf-8 --
import torch
from transformers import BertTokenizer, BertConfig
import torch.nn.functional as F
import numpy as np
from LawKeywordBert.models.bert_for_ner import BertSpanForNer
from LawKeywordBert.processors.ner_span import ner_processors as processors
from LawKeywordBert.processors.ner_span import collate_fn, InputFeature
from LawKeywordBert.processors.utils_ner import bert_extract_item
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch.nn as nn
from utils import get_cls_token
import networkx as nx
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise


class GNNClustering(nn.Module):
    def __init__(self, embedding_dim=768):
        super(GNNClustering, self).__init__()
        self.embedding_dim = embedding_dim

    @staticmethod
    def build_graph(embeddings):
        G = nx.Graph()
        num_nodes = embeddings.shape[0]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = pairwise.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1])
                if similarity > 0.5:
                    G.add_edge(i, j, weight=similarity.item())
                    print(i, j, similarity.item())
        return G

    @staticmethod
    def spectral_clustering(G, num_clusters=10):
        adjacency_matrix = nx.adjacency_matrix(G).todense()
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
        labels = clustering.fit_predict(adjacency_matrix)
        return labels


def diverse_keywords(
        word_embedding,
        words,
        top_n: int = 10,
        diverse_method: str = 'gnn'
):
    word_embedding = word_embedding.cpu().detach().numpy()

    if diverse_method == 'gnn':
        GNC = GNNClustering()
        keyword_graph = GNC.build_graph(word_embedding)
        labels = GNC.spectral_clustering(keyword_graph, num_clusters=top_n)
        print(words, labels)
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        results = []
        for label, indices in clusters.items():
            if len(indices) == 1:
                results.append(words[indices[0]])
            else:
                cluster_embeddings = word_embedding[indices]
                cluster_center = np.mean(cluster_embeddings, axis=0)

                distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)

                closest_node_index = indices[np.argmin(distances)]
                results.append(words[closest_node_index])

        return results

    else:
        raise ValueError('The {} is not supported.'.format(diverse_method))


def get_keyword(
        case: str,
        keywords: list,
        tokenizer: BertTokenizer,
        model: BertSpanForNer,
        config: BertConfig,
        k: int = 20,
        n: int = 10,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        diverse_method: str = 'msd'
):
    result = []
    candidate_word_embedding = []

    if len(keywords) == 0:
        return []

    word_tensor_db = []
    for word in keywords:
        word_cls = get_cls_token(word, tokenizer, model, config, device, for_word=True)
        word_tensor_db.append(word_cls)

    word_tensor_db = torch.stack(word_tensor_db).squeeze(1)

    doc_embedding = get_cls_token(case, tokenizer, model, config, device)
    doc_embedding = doc_embedding.expand(word_tensor_db.shape[0], 768)

    cosine_sim = F.cosine_similarity(word_tensor_db, doc_embedding, dim=-1)
    top_values, top_indices = torch.topk(cosine_sim, k=min(len(keywords), k))
    top_indices = top_indices.tolist()
    for value, idx in zip(top_values, top_indices):
        result.append((keywords[idx], round(value.item(), 4)))
        candidate_word_embedding.append(word_tensor_db[idx])

    candidate_words = [word[0] for word in result]
    keyword_result = diverse_keywords(
        torch.stack(candidate_word_embedding),
        candidate_words,
        top_n=n,
        diverse_method=diverse_method,
    )
    # print(keyword_result)
    return keyword_result


def sentence_based_truncation(
        case: str
) -> list:
    case_segment = []

    while len(case) >= 510:
        segment_pun = ['。', '；', ';', '，', ',']
        period_index = -1
        for pun in segment_pun:
            period_index = case[:510].rfind(pun)
            if period_index != -1:
                break

        case_segment.append(case[:period_index + 1])
        case = case[period_index + 1:]

    if len(case) >= 10:
        case_segment.append(case)

    return case_segment


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True, ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = " ".join(list(example))
        tokens = tokenizer.tokenize(textlist)
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            start_ids = ([0] * padding_length) + start_ids
            end_ids = ([0] * padding_length) + end_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_ids += ([0] * padding_length)
            end_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        features.append(InputFeature(input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=segment_ids,
                                     start_ids=start_ids,
                                     end_ids=end_ids,
                                     subjects=None,
                                     input_len=input_len))
    return features


def load_and_cache_examples(
        task,
        examples,
        tokenizer,
        max_seq_length: int = 512,
):
    processor = processors[task]()
    label_list = processor.get_labels()
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=max_seq_length,
                                            cls_token_at_end=False,
                                            pad_on_left=False,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=0,
                                            sep_token=tokenizer.sep_token,
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
                                            )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens)
    return dataset


def keyword_prediction(
        tokenizer: BertTokenizer,
        model: BertSpanForNer,
        config: BertConfig,
        task_name,
        examples: str,
        k: int = 20,
        n: int = 10,
        max_seq_length: int = 512,
        device: torch.device = torch.device('cpu'),
        method: str = 'ner',
        diverse_method: str = 'gnn',
) -> list:
    if method == 'ner':
        # split the examples so that each new examples are shorter than 510
        case_segment = sentence_based_truncation(examples)

        test_dataset = load_and_cache_examples(task_name, case_segment, tokenizer, max_seq_length)
        # print(len(test_dataset))

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)

        processor = processors[task_name]()
        label_list = processor.get_labels()
        id2label = {i: label for i, label in enumerate(label_list)}

        results = []
        for step, batch in enumerate(test_dataloader):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None, "end_positions": None,
                          "token_type_ids": batch[2]}
                outputs = model(**inputs)
            start_logits, end_logits = outputs[:2]
            R = bert_extract_item(start_logits, end_logits)
            if R:
                label_entities = [[id2label[x[0]], x[1], x[2]] for x in R]
            else:
                label_entities = []
            json_d = {'id': step, 'entities': label_entities}
            results.append(json_d)

        all_keywords = []
        for res, example in zip(results, case_segment):
            tag_dict = {'K': []}
            for entity in res['entities']:
                tag, start, end = entity
                if tag in tag_dict:
                    tag_dict[tag].append(example[start: end + 1])

            keywords = tag_dict['K']
            all_keywords.extend(keywords)

    else:
        raise ValueError('{} method is not supported.'.format(method))

    if all_keywords:
        all_keywords = get_keyword(examples, list(set(all_keywords)), tokenizer, model,
                                   config=config, k=k, n=n, device=device,
                                   diverse_method=diverse_method)
    else:
        all_keywords = ['无']
    # print(all_keywords)
    return all_keywords


if __name__ == '__main__':
    pass
