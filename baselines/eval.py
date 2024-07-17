from utils import load_model, write_to_json, load_json_file, design_prompt
import argparse
from evaluations import evaluation
import os
import torch
import numpy as np
import random


# Answer type of sketch dataset: Yes/No Question
# Answer type of vqa dataset: Yes/No Question and Digits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                    default="Qwen", 
                    choices=[
                        "mPLUG_Owl2", 
                        "InstructBLIP2", 
                        "InstructBLIP2-FlanT5-xl", 
                        "InstructBLIP2-FlanT5-xxl",
                        "InstructBLIP2-13B",
                        "InternLM",
                        "Qwen"])
    

    parser.add_argument("--dataset", type=str, 
                        default='ood-vqa',
                        choice=[
                            'ood-vqa',
                            'ood-vqa-challenge',
                            'sketch',
                            'sketch-challenge'
                        ])
    
    parser.add_argument("--custom_prompt", type=bool, default=False) 
    parser.add_argument("--save_jsonfile", type=str, default='./results/try1.json')
    parser.add_argument(
        "--precision", type=str, default='fp32', choices=['fp16', 'fp32']
    )

    # trial arguments
    parser.add_argument("--shots", nargs="+", default=[0], type=int)
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials to run for each shot using different demonstrations",
    )
    parser.add_argument(
        "--trial_seeds",
        nargs="+",
        default=[42],
        type=int,
        help="Seeds to use for each trial for picking demonstrations and eval sets",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)

    
    args = parser.parse_args()
    return args

def setup_seed(seed=16):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def main(args):

    # init model
    model = load_model(args.model_name)

    if args.dataset == 'ood-vqa':
        data_path = '../data/label/oodcv-vqa/oodcv-vqa.json'
    elif args.dataset == 'ood-vqa-challenge':
        data_path = '../data/label/oodcv-vqa/oodcv-counterfactual.json'
    elif args.dataset == 'sketch':
        data_path = '../data/label/sketchy-vqa/sketchy-vqa.json'
    elif args.dataset == 'sketch-challenge':
        data_path = '../data/label/sketchy-vqa/sketchy-challenging.json'
    
    data = load_json_file(data_path)
    img_paths = data['image']
    questions = data['question']

    if args.custom_prompt:
        prompt = design_prompt()
    else:
        prompt = ''

    preds = []

    for index, img_path in enumerate(img_paths):

        # inference
        pred = model.generate(
            instruction=[questions[index]],
            images=[img_path],
        )

        prediction = {
            'prediction': pred,
            'image_path': img_path
        }
        preds.append(prediction)


    # save submission file
    write_to_json(args.save_jsonfile, preds)

    # eval


if __name__ == "__main__":
    args = parse_args()
    main(args)