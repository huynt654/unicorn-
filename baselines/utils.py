import os
import sys
import json
import argparse


def process_exception(filename):
    flag = 0
    if filename == '.DS_Store':
        flag = 1
        return flag
    else: return flag

def load_model(TESTING_MODEL):

    if TESTING_MODEL == "mPLUG_Owl2": ## mmqa
        from mplug_owl2_modeling import  VLLMmPLUGOwl2
        model = VLLMmPLUGOwl2("MAGAer13/mplug-owl2-llama2-7b")

    elif TESTING_MODEL == "InstructBLIP2": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2("Salesforce/instructblip-vicuna-7b")
   
    elif TESTING_MODEL == "InstructBLIP2-13B": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2("Salesforce/instructblip-vicuna-13b")
   
    elif TESTING_MODEL == "InstructBLIP2-FlanT5-xl": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2("Salesforce/instructblip-flan-t5-xl")
   
    elif TESTING_MODEL == "InstructBLIP2-FlanT5-xxl": ## mmqa
        from instruct_blip_modeling import VLLMInstructBLIP2
        model = VLLMInstructBLIP2("Salesforce/instructblip-flan-t5-xxl")

    elif TESTING_MODEL == "Qwen": ## mmqa
        from qwen_vl_modeling import VLLMQwenVL
        model = VLLMQwenVL("Qwen/Qwen-VL-Chat")

    elif TESTING_MODEL == "InternLM": ## mmqa
        from internlm_xcomposer_modeling import VLLMInternLM
        model = VLLMInternLM(model_path="internlm/internlm-xcomposer-7b")
    
    return model

def load_json_file(file_path):

    """
    Load content in JSON file.

    Args:
        file_path (str): json path
    """

    try:

        with open(file_path, 'r') as file:
            configs = json.load(file)
        return configs
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        return None    

def design_prompt():
    return

def write_to_json(file_path, prediction):

    """
    Write prediction into file JSON.

    Args:
        file_path (str): Path to save
        prediction (dict): Model's prediction
    """

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(prediction, file, ensure_ascii=False, indent=4)
        print(f"Write prediction into {file_path} successfully")
    except Exception as e:
        print(f"Write prediction into {file_path} unsuccessfully")

def load_img_paths(base_dir):
    img_paths = []   # ../data
    sub_dirs = [os.path.join(base_dir, sub_dir) for sub_dir in os.listdir(base_dir)] # phase-1, phase-2

    for dir in sub_dirs:
        if process_exception(dir):
            continue
        for img_path in os.listdir(dir):
            if process_exception(img_path):
                continue
            img_paths.append(os.path.join(dir, img_path))

    return img_paths




print(len(os.listdir('/mnt/AI_Data/UNICORN/test/unicorn-/data/oodcv_images/phase-1/images')))


