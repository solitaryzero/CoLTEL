"""Training the CoLTEL model."""

import os
import json
import argparse
import time
import random
import pickle
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb
from peft import LoraConfig

from data import load_coltel_dataset, load_kilt_dictionary, process_naist_dictionary, process_naist_dataset
from utils import build_dataloader, get_constant_scheduler
from model import ColtelTokenizer, ColtelModel
from constants import full_special_tokens_map

def load_tuned_model(
    tuned_model_path,
    raw_model,
):
    custom_path = os.path.join(tuned_model_path, "custom_components.bin")
    if os.path.exists(custom_path):
        raw_model.load_state_dict(torch.load(custom_path), strict=False)

    for adapter_name in ["adapter_mention", "adapter_entity", "adapter_decoder"]:
        adapter_path = os.path.join(tuned_model_path, "adapters", adapter_name)
        if os.path.exists(adapter_path):
            raw_model.backbone.load_adapter(adapter_path, adapter_name=adapter_name)
    
    return raw_model

def run(
    args,
    entity_dict_dataset,
    test_dataset,
):
    """Run the training process."""
    if (args.report_to == 'wandb') and not(args.accelerate):
        wandb.init(
            project='Coltel',
            config={
                'base_model': args.base_model,
                'learning_rate': args.learning_rate,
                'seed_len': args.seed_len,
                'projector_type': args.projector_type,
                'lora_alpha': args.lora_alpha,
                'lora_r': args.lora_r,
                'epoch': args.epoch,
            }
        )

    llm = AutoModelForCausalLM.from_pretrained(args.base_model)

    if ('Qwen' in args.base_model):
        special_tokens_map = full_special_tokens_map['Qwen']
    elif ('Llama' in args.base_model):
        special_tokens_map = full_special_tokens_map['Llama']
    tokenizer = ColtelTokenizer.from_pretrained(args.base_model, special_tokens_map)

    if (args.recipe == 'entity'):
        input_type = 1
    elif (args.recipe == 'mention'):
        input_type = 0
    else:
        raise NotImplementedError

    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = pad_token_id
        llm.generation_config.pad_token_id = pad_token_id
    else:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.pad_token_id
        llm.generation_config.pad_token_id = pad_token_id

    lora_configs = {
        'decoder': LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
        ),
        'mention': LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
        ),
        'entity': LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
        ),
    }

    model = ColtelModel(
        llm=llm,
        tokenizer=tokenizer,
        lora_configs=lora_configs,
        projector_type=args.projector_type,
        seed_len=args.seed_len,
    )

    model = load_tuned_model(
        tuned_model_path=args.tuned_model_path,
        raw_model=model,
    )

    if args.fp16:
        model = model.half()
    elif args.bf16:
        model = model.bfloat16()

    entity_dict, id_to_name_map = process_naist_dictionary(
        examples=entity_dict_dataset, 
        latent_entity_token=tokenizer.latent_entity_token,
        seed_len=args.seed_len,
    )

    if (args.recipe == 'entity'):
        test_dataset, _ = process_naist_dictionary(
            examples=test_dataset, 
            latent_entity_token=tokenizer.latent_entity_token,
            seed_len=args.seed_len,
        )
    elif (args.recipe == 'mention'):
        test_dataset = process_naist_dataset(
            examples=test_dataset,
            id_to_name_map=id_to_name_map,
            latent_mention_token=tokenizer.latent_mention_token,
            seed_len=args.seed_len,
        )
    else:
        raise NotImplementedError

    correct, total = 0, 0
    all_predictions = []
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=args.max_new_tokens,
    )
    model.eval()
    model = model.to('cuda')

    for entry in tqdm(test_dataset, desc='Eval'):
        model_inputs = {
            'query': [entry['query']],
            'seed_tokens': [entry['seed_tokens']],
            'input_type': input_type,
        }
        prediction_outputs = model.predict(
            model_inputs,
            llm_generation_config=generation_config,
        )
        prediction = tokenizer.decode(prediction_outputs[0], skip_special_tokens=True)
        golden = entry['label']

        js = {
            'query': entry['query'],
            'golden': golden,
            'prediction': prediction,
        }
        all_predictions.append(js)

        if golden == prediction:
            correct += 1
        total += 1

    result = {
        'correct': correct,
        'total': total,
        'accuracy': correct/total,
    }
    if args.report_to == 'wandb':
        wandb.summary.update(result)

    return model, result, all_predictions

def main(args):
    """Main function entrance."""
    entity_dict_dataset = load_kilt_dictionary(data_path=args.dictionary_path)

    if (args.recipe == 'mention'):
        if 'kilt' in args.eval_data_path:
            test_split = 'validation'
        else:
            test_split = 'test'
        test_dataset = load_coltel_dataset(split=test_split, data_path=args.eval_data_path, shuffle=False)
    elif (args.recipe == 'entity'):
        shuffled_dataset = entity_dict_dataset.shuffle(seed=args.seed)
        test_dataset = shuffled_dataset.select(range(args.num_examples, args.num_examples+5000))
    else:
        raise NotImplementedError

    _model, result, predictions = run(
        args,
        entity_dict_dataset,
        test_dataset,
    )

    if result is not None:
        print('Accuracy:')
        print(result)

        os.makedirs(args.save_result_path, exist_ok=True)
        out_path = os.path.join(args.save_result_path, 'scores.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(result, fout)

        out_path = os.path.join(args.save_result_path, 'predictions.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(predictions, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path args
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--tuned_model_path', type=str, required=True)
    parser.add_argument('--dictionary_path', type=str, required=True)
    parser.add_argument('--eval_data_path', type=str, required=True)
    parser.add_argument('--save_result_path', type=str, required=True)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--recipe', choices=['entity', 'mention', 'sequential', 'mixed'])

    # Latent args
    parser.add_argument('--seed_len', type=int, default=1)
    parser.add_argument('--projector_type', choices=['linear', '2xMLP'], default='linear')

    # Lora args
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Misc
    parser.add_argument('--num_examples', type=int, default=200000)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report_to', type=str, default='none')

    _args = parser.parse_args()
    random.seed(_args.seed)
    torch.manual_seed(_args.seed)

    main(_args)
