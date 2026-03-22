from tqdm import tqdm

from datasets import load_dataset, load_from_disk, Dataset

from constants import prompt_templates, mention_start_token, mention_end_token

def load_coltel_dataset(
    split,
    data_path,
    shuffle=False,
    seed=42
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
        ds = load_dataset(data_path, split=split)
    else:
        ds = load_from_disk(data_path)[split]

    if shuffle:
        ds = ds.shuffle(seed=seed)

    return ds


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
        seed_tokens = latent_entity_token*seed_len,
        return {
            'id': example['id'],
            'label': example['name'],
            'description': example['description'],
            'query': query,
            'seed_tokens': seed_tokens,
        }
    
    dataset = examples.map(mapping_dictionary)
    dataset.add_faiss_index(column="id")

    return dataset


def retrieve_entity_by_id(
    dataset,
    target_id,
):
    scores, retrieved_examples = dataset.get_nearest_examples("id", target_id, k=1)
    
    return {
        "id": retrieved_examples["id"][0],
        'label': retrieved_examples['label'][0],
        "description": retrieved_examples["description"][0],
        "query": retrieved_examples["query"][0],
        "seed_tokens": retrieved_examples["seed_tokens"][0],
    }


def process_naist_dataset(
    examples,
    dictionary,
    latent_mention_token,
    seed_len=1,
):
    """Processes the dataset with naist-nlp format."""
    queries, seed_tokens, mentions, labels = [], [], [], []
    subsets, entry_ids = [], []
    iterator = tqdm(examples.iter(batch_size=1), desc='Preprocess')
    for entry in iterator:
        text = entry['text']
        subset = entry['subset']
        entry_id = entry['id']
        entities = entry['entity']

        for entity in entities:
            start, end, label = entity['start'], entity['end'], entity['label'][0]
            mention_text = text[start:end]
            entity = retrieve_entity_by_id(dictionary, label)

            print(mention_text)
            print(retrieve_entity_by_id(dictionary, label))
            input()

            augmented_text = text[:start] + mention_start_token + mention_text + mention_end_token + text[end:]
            query = prompt_templates['entity'].format(
                input_text=augmented_text,
                mention=mention_text,
            )
            seed_token = latent_mention_token*seed_len

            queries.append(query)
            seed_tokens.append(seed_token)
            mentions.append(mention_text)
            labels.append(entity['name'])
            subsets.append(subset)
            entry_ids.append(entry_id)

    processed_dataset = Dataset.from_dict({
        'query': queries,
        'seed_token': seed_tokens,
        'mention': mentions,
        'label': labels,
        'subset': subsets,
        'entry_id': entry_ids,
    })

    return processed_dataset