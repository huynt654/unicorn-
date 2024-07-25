import os
import sys
import json
import argparse
from llms.utils import respond 



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

    elif TESTING_MODEL == "QwenChat": ## mmqa
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

def design_prompt(prompt_type, question=None, description=None, context=None):
    
    '''

    question: str
    prompt_type: str
        - IP (Instruction Prompting)
        - IP-FS (Instruction Prompting - Few Shot)
        - CG (Context Generation)
        - QA: (Question Answering)
        - QA-FS: (Question Answering - Few Shot)
    description: str 
        A string text is returned by LLMs
    context: str
        A string text is returned by VLMs (step 1)
    
    return: str
        prompt 
    '''
    
    if prompt_type == 'IP':
        
        prompt = f'''Question: {question}
Task demonstration: What information needs to be described to help answer the question above? (describe in one sentence)
Answer:'''
  
    elif prompt_type == 'IP-FS':
        
        prompt = f'''Example 1: 
Question: 'How many people will dine at this table?'
Task demonstration: What information needs to be described to help answer the question above? (describe in one sentence)
Answer: A description of the table's size, shape, and the number of place settings or chairs arranged around it.

Example 2:
Question: 'What could block the washer's door?'
Task demonstration: What information needs to be described to help answer the question above? (describe in one sentence)
Answer: A description of the washer's surroundings, any visible obstructions near the door, and the current state of the washer's door (open or closed).

Example 3:
Question: "What is the hairstyle of the blond called?"
Task demonstration: What information needs to be described to help answer the question above? (describe in one sentence)
Answer: A description of the specific features or characteristics that distinguish the blond's hairstyle.

Question: {question}
Task demonstration: What information needs to be described to help answer the question above? (describe in one sentence)
Answer:'''
    
    elif prompt_type == 'CG':
        
        prompt = f'''{description}
        '''
    
    elif prompt_type == 'QA':
        
        prompt = f'''Context: {context}.
Question: {question}.
So the answer is: '''
    
    elif prompt_type == 'QA-FS':
        
        prompt = f'''Example 1:

Example 2:

Context: {context}.
Question: {question}.
Let's choose 1 of the options above. So the answer is: '''

    return prompt 

def load_step0(task_name, challenge):   
        if task_name == 'vqa':
            if challenge:
                # vqa-challenge
                jsonfile = '' 
            else: 
                # vqa
                jsonfile = ''
        else:
            if challenge:
                # sketchy-challenge
                jsonfile = ''
            else:
                # sketchy
                jsonfile = ''

        return jsonfile

def pipeline(vlm,
             llm_name,
             img_path, # list 
             question, # str 
             task_name, # str
             challenge, # int 
             step0=False
    ):

    '''

    Input:
    - vlm: model
    - llm_name: str
    - img_path: list[str]
    - question: str
    - step0: bool --> Whether to use step0 or not 
    Output:
    - pred: str (answer for VQA)

    '''
    
    
    if step0:
        description = ''
    else:
        jsonpath = load_step0(task_name, challenge)
        jsonfile = load_json_file(jsonpath)
        description = ''




    preds = vlm.generate()
    # outputs = eval_model.generate(
    #     images=batch_images,
    #     instruction=batch_text[0],
    # )
    return preds


def run_step0():
    
    # format: description: question
    
    return