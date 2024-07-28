import json

import os
import numpy as np
import random

from collections import defaultdict


import more_itertools
import torch

from evaluations.eval_datasets import (
    OODCVDataset,
    SketchyDataset
    )

from evaluations.sketchy_metric import get_sketchy_results
from evaluations.oodcv_vqa_metric import compute_oodcvqa_metrics
from tqdm import tqdm
from utils import pipeline

def setup_seed(seed=3407):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def test_ood_tasks(
    eval_model,
    llm_name,
    batch_size,
    image_dir_path,
    questions_json_path_test,
    seed=42,
    model_name=None,
    task_name=None,
    challenge=0
):
    setup_seed(seed)
    eval_dataset = OODCVDataset(
        image_base_dir_path=image_dir_path,
        question_path=questions_json_path_test,
    )

    def get_prompt_choose(sample, train=True):
        question = sample['question']
        if train:
            return f"{question}, choose from A or B\n{' '.join(sample['options'])}\nAnswer: {sample['answer']}"
        else:
            return f"{question}, choose from A or B\n{' '.join(sample['options'])}"
    
    def get_prompt_generate(sample, train=True):
        question = sample['question']
        if train:
            return f"{question} Short Answer: {sample['text_answer']}"
        else:
            return f"{question}"


    predictions = defaultdict()
    failed_case = 0
    inference = False

    for batch in more_itertools.chunked(tqdm(eval_dataset), batch_size):
        
        # according to batch
        batch_images, batch_text, batch_answers, situations = [], [], [], [] # image paths, prompt_text {prompt + question}, answer_text, situation 
        
        # iterative each element in batch
        for i in range(len(batch)):
            # add each element of batch to list 
            batch_images.append(batch[i]["image"]) # image
            batch_text.append(get_prompt_generate(batch[i], train=False)) # question 
            batch_answers.append(batch[i]["text_answer"]) # answer
            situations.append(batch[i]["situation"])
        
        # prediction in 1 sample 
        output = pipeline(eval_model, 
                          llm_name, 
                          batch_images, 
                          batch_text[0],
                          task_name, 
                          challenge,
                          inference
            )
        
        # outputs = eval_model.generate(
        #     images=batch_images,
        #     instruction=batch_text[0],
        # )

        for i, sample in enumerate(batch):
            predictions[sample["image"]] = {
                "prediction": output,
                "answer": batch_answers[0],
                "situation": situations[0],
            }


    if not os.path.exists("../test_results/generated/"):
        os.makedirs("../test_results/generated/", exist_ok=True)

    # save predictions
    if True:
            if challenge:
                save_path = f"../test_results/generated/{task_name}-counterfact_{model_name}_output.json"    
            else: 
                save_path = f"../test_results/generated/{task_name}_{model_name}_1st_output.json"
            with open(save_path, "w") as f:
                    json.dump(predictions, f, indent=4)
    
    counter_factual = True if challenge else False
    acc, yes_no_acc, digits_acc, all_digits_acc, all_category_results = compute_oodcvqa_metrics(predictions, counter_factual)
    
    print(f"Failed Case Number: {failed_case}..")
    
    
    return acc, yes_no_acc, digits_acc, all_digits_acc, all_category_results

def evaluate_sketchyvqa(
    eval_model,
    llm_name,
    batch_size,
    image_dir_path,
    questions_json_path,
    seed=42,
    model_name=None,
    task_name=None,
    challenge=0
):
    """
    Evaluate a model on POPE benchmark.
    """
    setup_seed(seed)
    full_dataset = SketchyDataset(
        image_base_dir_path=image_dir_path,
        question_path=questions_json_path,
    )

    inference = False
    predictions = []

    for batch in more_itertools.chunked(
        tqdm(full_dataset, desc="Running inference"), batch_size
    ):

        batch_images, batch_text = [], [] # image, question_text

        ## get example batch
        for i in range(len(batch)):
            batch_images.append([batch[i]["image"]])
            batch_text.append(batch[i]['question'])


        # prediction in 1 sample 
        outputs = pipeline(eval_model,
                            llm_name,
                            batch_images[0], 
                            batch_text[0],
                            task_name,
                            challenge, 
                            inference
            )
        # outputs = eval_model.generate(
        #     images=batch_images[0],
        #     instruction=batch_text[0],
        # )

        predictions.append(
                {"answer": outputs, "label": batch[0]["answer"], "image_path": batch[0]["image"]}
        )

    # save the predictions to a temporary file
    test_results_dir = "sketchyvqa_generated"
    if not os.path.exists(f"../test_results/{test_results_dir}/"):
        os.makedirs(f"../test_results/{test_results_dir}/", exist_ok=True)
       
    if challenge:
        out_file_path = f"../test_results/{test_results_dir}/{task_name}-counterfact_results_{model_name}.json"
    else:
        out_file_path = f"../test_results/{test_results_dir}/{task_name}_results_{model_name}.json"
    
    with open(out_file_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    _, metrics = get_sketchy_results(predictions)

    return metrics



