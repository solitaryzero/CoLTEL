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

# TODO: 先训entity，然后固定住decoder训mention

def save_model_no_accelerate(
    model,
    save_path,
):
    os.makedirs(save_path, exist_ok=True)

    model.backbone.save_pretrained(os.path.join(save_path, "adapters"))
    custom_state = {k: v.cpu() for k, v in model.state_dict().items() if "backbone" not in k}
    torch.save(custom_state, os.path.join(save_path, "custom_components.bin"))
    
    print(f"[Save] Model saved to {save_path}")

def save_model_accelerate(
    model,
    accelerator,
    save_path,
):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)

        unwrapped_model.backbone.save_pretrained(os.path.join(save_path, "adapters"))
        custom_state = {k: v.cpu() for k, v in unwrapped_model.state_dict().items() if "backbone" not in k}
        accelerator.save(custom_state, os.path.join(save_path, "custom_components.bin"))
        
        print(f"[Save] Model saved to {save_path}")

    accelerator.wait_for_everyone()

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
    train_dataset,
    test_dataset,
    save_model_path,
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

    if (args.tuned_model_path is not None):
        model = load_tuned_model(
            tuned_model_path=args.tuned_model_path,
            raw_model=model,
        )

    # Train
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
        train_dataset, _ = process_naist_dictionary(
            examples=train_dataset, 
            latent_entity_token=tokenizer.latent_entity_token,
            seed_len=args.seed_len,
        )
    elif (args.recipe == 'mention'):
        train_dataset = process_naist_dataset(
            examples=train_dataset,
            id_to_name_map=id_to_name_map,
            latent_mention_token=tokenizer.latent_mention_token,
            seed_len=args.seed_len,
        )

        # Load trained stage 1 params if possible
        if (args.tuned_model_path is not None):
            model = load_tuned_model(args.tuned_model_path, model)

            # Freeze the decoder to force mention adapter learn the information bottleneck
            model.set_adapter_trainable(
                adapter_name='adapter_decoder',
                trainable=False,
            )

    else:
        raise NotImplementedError

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        max_length=args.max_length,
        input_type=input_type,
    )

    optimizer = model.get_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_constant_scheduler(optimizer)

    # Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
    if (args.report_to == 'wandb') and (args.accelerate):
        accelerator.init_trackers(
            project_name='Coltel',
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

    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, scheduler
    )

    start_time = time.time()

    accumulated_loss = {
        'loss': 0.0,
        'ce_loss': 0.0,
        'mention_loss': 0.0,
        'entity_loss': 0.0,
    }
    accumulated_steps = 0
    for _epoch in tqdm(range(args.epoch), desc='Epoch'):
        for _step, batch in tqdm(enumerate(train_dataloader), desc='Step'):
            with accelerator.accumulate(model):
                outputs = model(
                    batch,
                )
                loss = outputs['loss']
                ce_loss = outputs['ce_loss']
                mention_loss = outputs['mention_loss']
                entity_loss = outputs['entity_loss']

                accumulated_steps += 1

                accumulated_loss['loss'] += loss.item()
                accumulated_loss['ce_loss'] += ce_loss.item()
                accumulated_loss['mention_loss'] += mention_loss.item()
                accumulated_loss['entity_loss'] += entity_loss.item()
                if accumulated_steps % args.logging_steps == 0:
                    avg_loss = accumulated_loss['loss'] / args.logging_steps
                    avg_ce_loss = accumulated_loss['ce_loss'] / args.logging_steps
                    avg_mention_loss = accumulated_loss['mention_loss'] / args.logging_steps
                    avg_entity_loss = accumulated_loss['entity_loss'] / args.logging_steps
                    if args.report_to == 'wandb':
                        if args.accelerate:
                            accelerator.log({
                                'train/loss': avg_loss,
                                'train/ce_loss': avg_ce_loss,
                                'train/mention_loss': avg_mention_loss,
                                'train/entity_loss': avg_entity_loss,
                            }, step=accumulated_steps)
                        else:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/ce_loss': avg_ce_loss,
                                'train/mention_loss': avg_mention_loss,
                                'train/entity_loss': avg_entity_loss,
                            }, step=accumulated_steps)
                    else:
                        print(f'Step {accumulated_steps} avg loss: {avg_loss}; CE: {avg_ce_loss}; Mention: {avg_mention_loss}; Entity: {avg_entity_loss}')

                    for key in accumulated_loss:
                        accumulated_loss[key] = 0.0

                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    tokenizer.save_pretrained(save_model_path)

    if args.accelerate:
        save_model_accelerate(
            model=model,
            accelerator=accelerator,
            save_path=save_model_path,
        )
    else:
        save_model_no_accelerate(
            model=model,
            save_path=save_model_path,
        )

    time_elapsed = time.time()-start_time
    run_time = time_elapsed

    h = int(run_time//3600)
    m = int((run_time-(h*3600))//60)
    s = run_time-(h*3600)-(m*60)
    print(f'[Training] Recipe {args.recipe} Run time : {h}h{m}m{s}s')

    if (args.do_eval) and (not(args.accelerate) or accelerator.is_main_process):
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
        if (args.accelerate):
            unwrapped_model = accelerator.unwrap_model(model)
        else:
            unwrapped_model = model

        for entry in tqdm(test_dataset, desc='Eval'):
            model_inputs = {
                'query': [entry['query']],
                'seed_tokens': [entry['seed_tokens']],
                'input_type': input_type,
            }
            prediction_outputs = unwrapped_model.predict(
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
            if args.accelerate:
                wandb_tracker = accelerator.get_tracker("wandb")
                wandb_tracker.run.summary.update(result)
            else:
                wandb.summary.update(result)

        accelerator.wait_for_everyone()
        accelerator.end_training()
        return model, result, all_predictions
    else:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return model, None, None

def main(args):
    """Main function entrance."""
    entity_dict_dataset = load_kilt_dictionary(data_path=args.dictionary_path)

    if (args.recipe == 'mention'):
        train_dataset = load_coltel_dataset(split='train', data_path=args.train_data_path, shuffle=True)
        train_dataset = train_dataset.select(range(args.num_examples))
        if 'kilt' in args.eval_data_path:
            test_split = 'validation'
        else:
            test_split = 'test'
        test_dataset = load_coltel_dataset(split=test_split, data_path=args.eval_data_path, shuffle=False)
    elif (args.recipe == 'entity'):
        shuffled_dataset = entity_dict_dataset.shuffle(seed=args.seed)
        train_dataset = shuffled_dataset.select(range(args.num_examples))
        test_dataset = shuffled_dataset.select(range(args.num_examples, 2*args.num_examples))
    else:
        raise NotImplementedError

    save_model_path = os.path.join(args.save_model_path)
    os.makedirs(save_model_path, exist_ok=True)
    _model, result, predictions = run(
        args,
        entity_dict_dataset,
        train_dataset,
        test_dataset,
        save_model_path,
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
    parser.add_argument('--tuned_model_path', type=str, default=None)
    parser.add_argument('--dictionary_path', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--eval_data_path', type=str, required=True)
    parser.add_argument('--save_model_path', type=str)
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

    # Training args
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=2000)

    # Misc
    parser.add_argument('--num_examples', type=int, default=200000)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report_to', type=str, default='none')
    parser.add_argument('--accelerate', action='store_true')

    _args = parser.parse_args()
    random.seed(_args.seed)
    torch.manual_seed(_args.seed)

    main(_args)
