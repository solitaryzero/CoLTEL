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
from accelerate import Accelerator
import wandb

from data import load_colt_dataset, process_gsm8k_aug
from utils import build_dataloader, get_constant_scheduler
from model import ColtelTokenizer, ColtelModel


def run(
    args,
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
    tokenizer = ColtelTokenizer.from_pretrained(args.base_model)

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

    lora_configs = {}

    model = ColtelModel(
        llm=llm,
        tokenizer=tokenizer,
        lora_configs=lora_configs,
        projector_type='2xMLP',
        seed_length=args.seed_len,
    )

    # Train
    if args.fp16:
        model = model.half()
    elif args.bf16:
        model = model.bfloat16()

    train_dataset = process_gsm8k_aug(
        train_dataset,
        step_latent_token=tokenizer.latent_trigger_token,
        step_latent_token_id=tokenizer.latent_trigger_token_id,
        step_seperator_token=tokenizer.latent_sep_token,
        latent_end_token=tokenizer.latent_end_token,
        seed_token=tokenizer.latent_seed_token,
        seed_len=args.seed_len,
        test_mode=False,
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        latent_token_id_list=latent_token_id_list,
    )

    optimizer = model.get_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_constant_scheduler(optimizer)

    # Accelerate
    accelerator = Accelerator(log_with="wandb")
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
        'latent_loss': 0.0,
    }
    accumulated_steps = 0
    for _epoch in tqdm(range(args.epoch), desc='Epoch'):
        for _step, batch in tqdm(enumerate(train_dataloader), desc='Step'):
            with accelerator.accumulate(model):
                outputs = model(
                    batch,
                )
                loss = outputs['loss']
                ce_loss, latent_loss = outputs['ce_loss'], outputs['latent_loss']

                accumulated_steps += 1

                accumulated_loss['loss'] += loss.item()
                accumulated_loss['ce_loss'] += ce_loss.item()
                accumulated_loss['latent_loss'] += latent_loss.item()
                if accumulated_steps % args.logging_steps == 0:
                    avg_loss = accumulated_loss['loss'] / args.logging_steps
                    avg_ce_loss = accumulated_loss['ce_loss'] / args.logging_steps
                    avg_latent_loss = accumulated_loss['latent_loss'] / args.logging_steps
                    if args.report_to == 'wandb':
                        if args.accelerate:
                            accelerator.log({
                                'train/loss': avg_loss,
                                'train/ce_loss': avg_ce_loss,
                                'train/latent_loss': avg_latent_loss,
                            }, step=accumulated_steps)
                        else:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/ce_loss': avg_ce_loss,
                                'train/latent_loss': avg_latent_loss,
                            }, step=accumulated_steps)
                    else:
                        print(f'Step {accumulated_steps} avg loss: {avg_loss}; CE: {avg_ce_loss}; Latent: {avg_latent_loss}')

                    for key in accumulated_loss:
                        accumulated_loss[key] = 0.0

                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    tokenizer.save_pretrained(save_model_path)

    if args.accelerate:
        accelerator.save_model(model, save_model_path)
        model_config = {
            'base_model': args.base_model,
            'unpacker_init_params': unpacker_init_params,
            'max_latent_steps': args.max_latent_steps,
            'probe_layer': args.probe_layer,
            'max_seed_len': args.seed_len,
        }
        with open(os.path.join(save_model_path, 'model_config.bin'), 'wb') as fout:
            pickle.dump(model_config, fout)
    else:
        model_file_name = os.path.join(save_model_path, f'{args.latent_type}_colt.bin')
        torch.save(model, model_file_name)

    time_elapsed = time.time()-start_time
    run_time = time_elapsed

    h = int(run_time//3600)
    m = int((run_time-(h*3600))//60)
    s = run_time-(h*3600)-(m*60)
    print(f'[Training] Run time : {h}h{m}m{s}s')

    # Eval
    if (args.do_eval) and (not(args.accelerate) or accelerator.is_main_process):
        test_dataset = process_gsm8k_aug(
            test_dataset,
            step_latent_token=tokenizer.latent_seed_token,
            step_latent_token_id=tokenizer.latent_seed_token_id,
            step_seperator_token=tokenizer.latent_sep_token,
            latent_end_token=tokenizer.latent_end_token,
            test_mode=True,
        )

        if args.fp16:
            model = model.half()
        elif args.bf16:
            model = model.bfloat16()

        correct, total = 0, 0
        total_cot_tokens = 0
        generation_config = GenerationConfig(
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        all_predictions = []
        for entry in tqdm(test_dataset, desc='Eval'):
            model_inputs = {
                'question': [entry['query']],
            }

            with torch.no_grad():
                if args.accelerate:
                    latent_generation_outputs = accelerator.unwrap_model(model).generate(
                        model_inputs,
                        llm_generation_config=generation_config,
                        return_dict_in_generate=True,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
                    latent_generation_outputs = model.generate(
                        model_inputs,
                        llm_generation_config=generation_config,
                        return_dict_in_generate=True,
                        max_new_tokens=args.max_new_tokens,
                    )
                generation_outputs = latent_generation_outputs['prediction']
                num_latents = latent_generation_outputs['num_latents']

            decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
            golden = entry['answer'].strip().split('The answer is ')[-1].strip('.').replace(tokenizer.eos_token, '')
            golden = float(golden.replace(',', '').replace(' ', ''))
            try:
                prediction = decoded.strip().split('The answer is ')[-1].strip('.')
                prediction = float(prediction.replace(',', '').replace(' ', ''))
            except (ValueError, TypeError, OverflowError) as _e:
                prediction = 'Dummy prediction'

            total_cot_tokens += num_latents[0].item()

            js = {
                'query': entry['query'],
                'output': decoded,
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
            'avg_tokens': total_cot_tokens/total,
        }
        if args.report_to == 'wandb':
            if args.accelerate:
                wandb_tracker = accelerator.get_tracker("wandb")
                wandb_tracker.run.summary.update(result)
            else:
                wandb.summary.update(result)

        accelerator.end_training()
        return model, result, all_predictions
    else:
        accelerator.end_training()
        return model, None, None

def main(args):
    """Main function entrance."""
    train_dataset = load_colt_dataset(split='train', data_path=args.data_path, shuffle=True)
    test_dataset = load_colt_dataset(split='test', data_path=args.data_path, shuffle=False)

    save_model_path = os.path.join(args.save_model_path)
    os.makedirs(save_model_path, exist_ok=True)
    _model, result, predictions = run(args, train_dataset, test_dataset, save_model_path)

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
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--result_probe_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--save_result_path', type=str, required=True)
    parser.add_argument('--accelerator_config', type=str, default=None)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    # Latent args
    parser.add_argument('--latent_type', type=str, required=True)
    parser.add_argument('--unpacker_layer_num', type=int, default=2)
    parser.add_argument('--seed_len', type=int, default=1)
    parser.add_argument('--intermediate_dim', type=int, default=4096)
    parser.add_argument('--max_latent_steps', type=int, default=10)
    parser.add_argument('--int_precision', type=int, default=10)
    parser.add_argument('--frac_precision', type=int, default=5)
    parser.add_argument('--probe_layer', type=int, default=-1)
    parser.add_argument('--projector_type', choices=['linear', '2xMLP'], default='linear')

    # Lora args
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_r', type=int, default=128)

    # Training args
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--train_latent_projector', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=2000)

    # Misc
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report_to', type=str, default='none')
    parser.add_argument('--accelerate', action='store_true')

    _args = parser.parse_args()
    random.seed(_args.seed)
    torch.manual_seed(_args.seed)

    main(_args)
