import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR


class ColtelCollator:
    def __init__(self, tokenizer=None, max_length=2048, input_type=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_type = input_type

    def tokenize_qa_pair(
        self,
        questions,
        answers,
    ):
        full_texts = [
            q + a + self.tokenizer.eos_token
            for q, a in zip(questions, answers)
        ]

        tokenized_full_text = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
        )

        input_ids = tokenized_full_text.input_ids
        attention_mask = tokenized_full_text.attention_mask

        tokenized_questions = self.tokenizer(
            questions,
            padding=False,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
        )

        question_lengths = [len(x) for x in tokenized_questions.input_ids]
        labels = input_ids.clone()
        for i, l in enumerate(question_lengths):
            labels[i, :l] = -100
        
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    def __call__(self, elements):
        queries = [e['query'] for e in elements]
        answers = [e['seed_tokens'] for e in elements]
        tokenize_results = self.tokenize_qa_pair(queries, answers)

        decoder_texts = [(e['label'] + self.tokenizer.eos_token) for e in elements]
        decoder_tokenize_results = self.tokenizer(
            decoder_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length,
        )
        decoder_input_ids = decoder_tokenize_results['input_ids']
        decoder_attention_mask = decoder_tokenize_results['attention_mask']
        decoder_labels = decoder_input_ids.clone()
        decoder_labels[(decoder_attention_mask == 0)] = -100

        batch = {
            'input_ids': tokenize_results['input_ids'],
            'labels': tokenize_results['labels'],
            'attention_mask': tokenize_results['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'decoder_labels': decoder_labels,
            'decoder_attention_mask': decoder_attention_mask,
            "input_type": self.input_type,
        }

        return batch


def build_dataloader(
    dataset,
    batch_size,
    tokenizer,
    **kwargs,
):
    collator = ColtelCollator(tokenizer=tokenizer, **kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
    )

    return dataloader