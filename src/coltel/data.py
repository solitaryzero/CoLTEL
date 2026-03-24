import os
from tqdm import tqdm

from datasets import load_dataset, load_from_disk, Dataset

from constants import prompt_templates, mention_start_token, mention_end_token

def load_coltel_dataset(
    split,
    data_path,
    shuffle=False,
    seed=42,
):
    """Loads a dataset from huggingface hub or local storage."""
    hf_dataset_identifiers = set([
        "naist-nlp/kilt",
        "naist-nlp/wned-wiki",
        "naist-nlp/wned-cweb",
        "naist-nlp/msnbc",
        "naist-nlp/aquaint",
        "cyanic-selkie/aida-conll-yago-wikidata",
    ])

    if data_path in hf_dataset_identifiers:
        if ('kilt' in data_path):
            ds = load_dataset(data_path, 'dataset', split=split)
        else:
            ds = load_dataset(data_path, split=split)
    else:
        if ('kilt' in data_path):
            ds = load_from_disk(os.path.join(data_path, 'dataset'))[split]
        else:
            ds = load_from_disk(data_path)[split]

    if shuffle:
        ds = ds.shuffle(seed=seed)

    return ds

def load_kilt_dictionary(
    data_path,
):
    hf_dataset_identifiers = set([
        "naist-nlp/kilt",
    ])

    if data_path in hf_dataset_identifiers:
        dictionary = load_dataset(data_path, 'dictionary', split='kb')
    else:
        dictionary = load_from_disk(os.path.join(data_path, 'dictionary'))['kb']

    return dictionary

def process_naist_dictionary(
    examples,
    latent_entity_token=None,
    seed_len=1,
):
    """Processes the wikipedia dictionary with naist-nlp format."""
    def mapping_dictionary(example):
        query = prompt_templates['entity'].format(
            description_text=example['description'],
        )
        seed_tokens = latent_entity_token*seed_len

        return {
            'id': example['id'],
            'label': example['name'],
            'description': example['description'],
            'query': query,
            'seed_tokens': seed_tokens,
        }
    
    dataset = examples.map(mapping_dictionary)
    id_to_name_map = {eid: ename for eid, ename in zip(dataset["id"], dataset["label"])}

    return dataset, id_to_name_map


def process_naist_dataset(
    examples,
    id_to_name_map,
    latent_mention_token,
    seed_len=1,
):
    """Processes the dataset with naist-nlp format."""
    def mapping_mentions(batch):
        batch_size = len(batch['id'])
        queries, seed_tokens, mentions, labels = [], [], [], []
        subsets, entry_ids = [], []
        for i in range(batch_size):
            text = batch['text'][i]
            subset = batch['subset'][i]
            entry_id = batch['id'][i]
            entities = batch['entities'][i]

            for entity in entities:
                start, end, label = entity['start'], entity['end'], entity['label'][0]
                mention_text = text[start:end]
                retrieved_entity = id_to_name_map[label]

                augmented_text = text[:start] + mention_start_token + mention_text + mention_end_token + text[end:]
                query = prompt_templates['mention'].format(
                    input_text=augmented_text,
                    mention=mention_text,
                )
                seed_token = latent_mention_token*seed_len

                queries.append(query)
                seed_tokens.append(seed_token)
                mentions.append(mention_text)
                labels.append(retrieved_entity)
                subsets.append(subset)
                entry_ids.append(entry_id)

        return {
            'query': queries,
            'seed_tokens': seed_tokens,
            'mention': mentions,
            'label': labels,
            'subset': subsets,
            'entry_id': entry_ids,
        }

    processed_dataset = examples.map(
        mapping_mentions,
        batched=True,
        remove_columns=examples.column_names,
    )

    return processed_dataset