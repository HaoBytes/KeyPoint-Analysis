# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used to finetune the Flan-T5-3B/11B for KPG.
"""

import json, logging
import argparse
from functools import partial
import os, math
import sys
import nltk
import numpy as np
import pandas as pd
import random as rn
from tqdm.auto import tqdm

import datasets
import evaluate

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    WeightedRandomSampler, 
    SequentialSampler, 
    TensorDataset, 
    Dataset
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)
import wandb


import regex as re
import random as rn
import datasets

from rouge_setbase import preprocess_dataset, compute_rouge
from softF1 import softevaluation
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, average_precision_score, precision_score
#from tqdm.notebook import tqdm
#from tqdm import trange

#from typing import Optional
#from dataclasses import dataclass, field


logger = get_logger(__name__)

def load_dataset(input_file, input_val_file, input_test_file, val=True) -> pd.DataFrame:
    # Create a new dataframe with only the rows of 'data' that don't match the 'Summary' column of 'test_df'

    # Load the CSV file into a pandas dataframe
    train_df = pd.read_csv(input_file)
    val_df = pd.read_csv(input_val_file)
    test_df = pd.read_csv(input_test_file)
    
    df_train = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    df_val = val_df.loc[:, ~val_df.columns.str.contains('^Unnamed')]
    df_test = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

    df_train['stance_list_arg_topic'] = df_train['stance'].astype(str) + df_train['list_arg_topic']
    df_val['stance_list_arg_topic'] = df_val['stance'].astype(str) + df_val['list_arg_topic']
    df_test['stance_list_arg_topic'] = df_test['stance'].astype(str) + df_test['list_arg_topic'] 
    df_train = df_train.drop(columns=['key_point_topic'])

    if val:
        df_train = df_train
    else:
        combined_df = pd.concat([df_train, df_val], ignore_index=True)
        # Reset the index of the combined dataframe
        df_train = combined_df.reset_index(drop=True)

    # Convert the pandas dataframes to Hugging Face datasets
    train_dataset = Dataset.from_pandas(df_train)
    valid_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_dict(df_test)

    # Create a DatasetDict object that contains the train, validation and test datasets
    my_dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": valid_dataset})

    return my_dataset_dict


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["stance_list_arg_topic"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    print(type(examples))
    try:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["key_point"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
    except KeyError:
        pass
    return model_inputs


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    if metric_name == "rouge":
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    else:
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True, help="Path to a csv file into model")
    parser.add_argument("--input_test_file", type=str, required=False, help="Path to a test csv file into model")
    parser.add_argument("--input_val_file", type=str, required=False, help="Path to a val csv file into model")
    parser.add_argument("--output_file", type=str, default="system.txt", help="Generate sample")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to an output directory were the finetuned model and results are saved",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the pretrained model to use for finetuning",
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to the model directory if using a locally saved model"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="google/flan-t5-base", help="Hugging Face tokenizer name"
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="The training batch size per GPU")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="The eval batch size per GPU")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="The number of gradient accumulation steps to perform",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The maximum learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="The maximum number of training epochs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2023],
        help="The seeds to use. If multiple are given, the entire finetuning process is repeated multiple times.",
    )
    parser.add_argument("--max_input_length", type=int, default=512, help="Maximum length of input sequence")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximum length of target sequence")
    parser.add_argument(
        "--predict_with_generate", action="store_true", help="Whether to use generation for prediction."
    )
    parser.add_argument("--a100", action="store_true", help="Use BF16 and TF32.")
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true", help="Test mode")    
    parser.add_argument("--prefix", 
                        type=str, 
                        default="summarize: ", 
                        help="prefix for flan-T5")
    
    parser.add_argument("--bleurt", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bartscore", action="store_true")  

    args = parser.parse_args()
    val = not args.no_val
    assert args.input_test_file or args.input_val_file, "Do you want to train only? Really?"
    args.input_test_file = args.input_test_file or args.input_val_file
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    prefix = args.prefix

    # some hard-coded parameters
    args.max_train_steps = None
    args.num_beams = 6
    args.with_tracking = False
    args.num_warmup_steps = 0
    args.lr_scheduler_type = "linear"
    args.resume_from_checkpoint = None

    bleurt = args.bleurt
    bertscore = args.bertscore
    bartscore = args.bartscore

    # preprocess_function = partial(
    #     preprocess_function, max_input_length=max_input_length, max_target_length=max_target_length
    # )

    print('model_path is ', args.model_path)
    if args.model_path:
        # print('Loading model from local path ', args.model_path)
        logger.info('Loading model from local path ', args.model_path)
        model_path = args.model_path
        if 'pytorch_model.bin' in os.listdir(model_path) and args.model_name is not None:
            config = AutoConfig.from_pretrained(args.model_name)
            model = AutoModelForSeq2SeqLM.from_config(config)
            model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu'))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        # print('Done with loading model from local path.')
        logger.info('Done with loading model from local path.')
    else:
        model_name = args.model_name
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Metric
    metric = evaluate.load("rouge")

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = 'wnadb'
        accelerator_log_kwargs["logging_dir"] = args.output_dir
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    for seed in args.seeds:
        set_seed(seed)
        rn.seed(seed)
        output_dir = os.path.join(args.output_dir, str(seed))

        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # Load the dataset using the load_dataset function


        # input_file, input_val_file, input_test_file, input_dg_file=None, val=True
        my_dataset_dict = load_dataset(
            input_file=args.input_file,
            input_val_file=args.input_val_file,
            input_test_file=args.input_test_file,
            val=val
        )
        with accelerator.main_process_first():
            train_dataset = my_dataset_dict['train'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['train'].column_names, load_from_cache_file=False)
            eval_dataset = my_dataset_dict['validation'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['validation'].column_names, load_from_cache_file=False)
            test_dataset = my_dataset_dict['test'].map(preprocess_function, batched=True, remove_columns=my_dataset_dict['test'].column_names, load_from_cache_file=False)
        
        # Log a few random samples from the training set:
        if args.train:
            for index in rn.sample(range(len(train_dataset)), 1):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        if args.test:
            for index in rn.sample(range(len(test_dataset)), 1):
                logger.info(f"Sample {index} of the testing set: {test_dataset[index]}.")
        

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8 if accelerator.use_fp16 else None)

        train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # prepare model before optimizer
        model, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
                model, train_dataloader, eval_dataloader, test_dataloader
            )
        
        # to make model be able to be evaluated if wrapped by FSDP mode
        dummy_inputs = tokenizer(
            'This is a dummy input for the purpose of FSDP wrapping',
            text_target = "OK, ignored.",
            max_length=args.max_input_length, padding=False, truncation=True, 
            return_tensors='pt'
        )

        if args.train:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

            overrode_max_train_steps = False
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            if args.max_train_steps is None:
                args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
                overrode_max_train_steps = True

            lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            )

            optimizer, lr_scheduler = accelerator.prepare(
                optimizer, lr_scheduler
            )

            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            if overrode_max_train_steps:
                args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            # Afterwards we recalculate our number of training epochs
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

            if args.with_tracking:
                experiment_config = vars(args)
                # TensorBoard cannot log Enums, need the raw value
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
                accelerator.init_trackers("pretrain_t5_no_trainer", experiment_config)
            
            # Train!
            total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {args.max_train_steps}")
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
            completed_steps = 0
            starting_epoch = 0

            if args.resume_from_checkpoint:
                if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                    accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                    accelerator.load_state(args.resume_from_checkpoint)
                    path = os.path.basename(args.resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                    dirs.sort(key=os.path.getctime)
                    path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                else:
                    resume_step = int(training_difference.replace("step_", ""))
                    starting_epoch = resume_step // len(train_dataloader)
                    resume_step -= starting_epoch * len(train_dataloader)
            
            for epoch in range(starting_epoch, args.num_train_epochs):
                model.train()
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    # We need to skip steps until we reach the resumed step
                    if args.resume_from_checkpoint and epoch == starting_epoch:
                        if resume_step is not None and step < resume_step:
                            completed_steps += 1
                            continue

                    with accelerator.accumulate(model):
                        outputs = model(**batch)
                        loss = outputs.loss
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            total_loss += loss.detach().float()
                            accelerator.log({"train_step_loss": loss.detach().float()}, step=completed_steps)
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        completed_steps += 1

                
                # save training state for each epoch
                # accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch}"))
                    
            # save final-epoch model
            accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch}"))

            # accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(
            #     os.path.join(output_dir, f"epoch_{epoch}"), is_main_process=accelerator.is_main_process, 
            #     save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
            # )
              
            # do validation
            if val:
                model.eval()
                gen_kwargs = {
                    "max_length": max_target_length,
                    "num_beams": args.num_beams,
                    "synced_gpus": True,
                }
                # run dummy inputs
                _ = accelerator.unwrap_model(model)(**dummy_inputs)
                epoch_predictions, epoch_labels = [], []
                for e_step, e_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            e_batch["input_ids"],
                            attention_mask=e_batch["attention_mask"],
                            **gen_kwargs,
                        )
                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = e_batch["labels"]
                        labels = accelerator.pad_across_processes(e_batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                        generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                        generated_tokens = generated_tokens.cpu().numpy()
                        labels = labels.cpu().numpy()
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]
                            
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        
                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                        epoch_predictions += decoded_preds
                        epoch_labels += decoded_labels
                        metric.add_batch(
                            predictions=decoded_preds,
                            references=decoded_labels,
                        )
                
                result = metric.compute(use_stemmer=True)
                result = {k: round(v * 100, 4) for k, v in result.items()}

                logger.info(f"Eval results at epoch {epoch}: {result}")

                # save the predictions
                with open(os.path.join(output_dir, f"predictions_{epoch}.json"), "w") as pf:
                    for p in [{"prediction" : k, "label": v} for k, v in zip(epoch_predictions, epoch_labels)]:
                        pf.write(json.dumps(p) + "\n")
                

        # do test
        if args.test:
            logger.info("***** Running testing *****")
            logger.info(f"  Num examples = {len(test_dataset)}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
            model.eval()
            model.config.use_cache = True
            gen_kwargs = {
                "max_length": max_target_length,
                "num_beams": args.num_beams,
                "synced_gpus": True,
            }

             # run dummy inputs
            _ = accelerator.unwrap_model(model)(**dummy_inputs)
            test_predictions = []
            for t_step, t_batch in enumerate(test_dataloader):
                generated_tokens = accelerator.unwrap_model(model).generate(
                    t_batch["input_ids"],
                    attention_mask=t_batch["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = accelerator.gather_for_metrics(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()
                print(generated_tokens)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_preds = [pred.strip() for pred in decoded_preds]
                test_predictions += decoded_preds

            # save the predictions
            output_test_preds_file = os.path.join(output_dir, args.output_file)
            output_test_preds_file = output_test_preds_file.replace('.txt', '.csv') # replace .txt with .csv

            with open(output_test_preds_file, "w") as writer:
                for pred in test_predictions:
                    json.dump(pred, writer) # write each prediction as a JSON object on a separate line
                    writer.write('\n') # add a newline character to separate each object


            #compute the set-base metrics of rouge , bertsecore, bleurt and bartscore
            pd.DataFrame(list(zip(range(0,len(test_dataset)), test_predictions, my_dataset_dict['test']['key_point'], my_dataset_dict['test']['topic'], my_dataset_dict['test']['stance'])), columns=['ID', 'key_point', 'key_point_given', 'topic', 'stance']).to_csv(output_test_preds_file, index=False)

            gt_gold_kp = pd.read_csv(output_test_preds_file)
            predictions, references = preprocess_dataset(gt_gold_kp)
            compute_rouge(predictions, references)

            if bleurt:
                bleurt_p = softevaluation(gt_gold_kp, "precision", "BLEURT")
                bleurt_r = softevaluation(gt_gold_kp, "precision", "BLEURT")
                bleurt_f1 = softevaluation(gt_gold_kp, "f1", "BLEURT")
                print(f"Precision of BLEURT is {bleurt_p}; Recall of BLEURT is {bleurt_r}; Soft F1 of BLEURT is {bleurt_f1}")

            if bertscore:
                bertscore_p = softevaluation(gt_gold_kp, "precision", "BERTScore")
                bertscore_r = softevaluation(gt_gold_kp, "precision", "BERTScore")
                bertscore_f1 = softevaluation(gt_gold_kp, "f1", "BERTScore")
                print(f"Precision of BERTSCORE is {bertscore_p}; Recall of BERTSCORE is {bertscore_r}; Soft F1 of BERTSCORE is {bertscore_f1}")
            
            if bartscore:
                bartscore_p = softevaluation(gt_gold_kp, "precision", "BARTScore")
                bartscore_r = softevaluation(gt_gold_kp, "precision", "BARTScore")
                bartscore_f1 = softevaluation(gt_gold_kp, "f1", "BARTScore")
                print(f"Precision of BERTSCORE is {bartscore_p}; Recall of BERTSCORE is {bartscore_r}; Soft F1 of BERTSCORE is {bartscore_f1}")
