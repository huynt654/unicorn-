from utils import load_model, write_to_json, design_prompt
import argparse
import os
import torch
import numpy as np
import random

from evaluations.evaluation import test_ood_tasks, evaluate_sketchyvqa

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
                            'sketch-challenge',
                            'ok-vqa',
                            'a-okvqa',
                            'gqa-testdev'
                        ])
    parser.add_argument("--save_jsondir", type=str, default='./results/')
        # trial arguments
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials to run for each shot using different demonstrations",
    )
    parser.add_argument(
        "--trial_seeds",
        nargs="+",
        default=[16],
        type=int,
        help="Seeds to use for each trial for picking demonstrations and eval sets",
    )

    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--custom_prompt", type=bool, default=False) 

    args = parser.parse_args()
    return args

def main(args):

    # init model
    model = load_model(args.model_name).to(args.deivce)

    # determine dataset
    if args.dataset == 'ood-vqa':
        question_jsonfile = '../data/label/oodcv-vqa/oodcv-vqa.json'
        eval_task = 'vqa'
        img_base_dir = '../data/oodcv_images'
    elif args.dataset == 'ood-vqa-challenge':
        question_jsonfile = '../data/label/oodcv-vqa/oodcv-counterfactual.json'
        eval_task = 'vqa'
        img_base_dir = '../data/oodcv_images'
    elif args.dataset == 'sketch':
        question_jsonfile = '../data/label/sketchy-vqa/sketchy-vqa.json'
        eval_task = 'sktechy'
        img_base_dir = '../data/sketchy_images'
    elif args.dataset == 'sketch-challenge':
        question_jsonfile = '../data/label/sketchy-vqa/sketchy-challenging.json'
        eval_task = 'sktechy'
        img_base_dir = '../data/sktechy_images'


    final_results = {}
    # muốn viết eval cho ood-vqa, sketchy
    if eval_task == 'vqa':
        acc_list, yes_no_acc_list, digits_acc_list, all_digits_acc_list = [], [], [], []

        for seed in args.trial_seeds:
            acc, yes_no_acc, digits_acc, all_digits_acc, all_category_results = test_ood_tasks(
                model, 
                args.batch_size,

                img_base_dir,
                question_jsonfile,
                
                seed=seed,
                model_name=args.model_name,
                task_name=eval_task,
            )


            acc_list.append(acc)
            yes_no_acc_list.append(yes_no_acc)
            digits_acc_list.append(digits_acc)
            all_digits_acc_list.append(all_digits_acc)
            
            # print results
            print("Acc.: {:.2f}; Yes/No Acc.: {:.2f}; Digits Acc.: {:.2f}".format(acc, yes_no_acc, digits_acc))
        
        final_results[args.model_name] = {
            "Acc": acc_list,
            "DigitsAcc": digits_acc_list,
            "YesNoAcc": yes_no_acc_list,
            "AllDigits": all_digits_acc_list[0],
            "AllCategories": all_category_results,
        }
    elif args.eval_sketch or args.eval_sketch_challenging:
        
        scores = evaluate_sketchyvqa(
            model,
            batch_size=args.batch_size,
            
            
            image_dir_path=img_base_dir,
            questions_json_path=question_jsonfile,
            
            
            seed=args.trial_seeds[0],
            model_name=args.model_name,
            task_name=eval_task,
        )

        final_results[args.model_name] = [{
                "acc": scores[0],
                "precision": scores[1],
                "recall": scores[2],
                "f1": scores[3],
                "yes_ratio": scores[4],
            }]
        print(final_results)

    # save result file
    json_file = os.path.join(args.save_jsondir, f'{args.dataset}_{args.model_name}.json')
    write_to_json(json_file, final_results)


if __name__ == "__main__":
    args = parse_args()
    main(args)