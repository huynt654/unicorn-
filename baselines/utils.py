import json
from llms.utils import respond 
import os
from tqdm import tqdm 

class DescriptionNotFoundError(Exception):
    'Exception was initialized when not dound corresponding description'
    pass

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
Question: {question}
So the answer is: '''
    
    elif prompt_type == 'QA-FS':
        
        prompt = f'''Example 1:

Example 2:

Context: {context}.
Question: {question}.
Let's choose 1 of the options above. So the answer is: '''

    return prompt 

def get_description(data, target_question, target_image_path):
    for item in data:
        if item['question'] == target_question and item['image_path'] == target_image_path:
            return item['description']
    raise DescriptionNotFoundError(f"KNot found description for question '{target_question}' and image '{target_image_path}'")

def pipeline(vlm,
             llm_name,
             img_path, # list 
             question, # str 
             task_name, # str
             challenge, # int
             inference, # bool
             desc=None, # str 
             step0=False
    ):

    '''

    Input:
    - vlm: model
    - llm_name: str
    - img_path: list[str]
    - question: str
    - inference: bool --> Whether print prediction or not 
    - desc: str 
    - step0: bool --> Whether to use step0 or not 
    Output:
    - pred: str (answer for VQA)

    '''
     
    if step0:
        # Step 0: Generate description llm
        IP_FS_prompt = design_prompt('IP-FS', question)
        description = respond(llm_name,  IP_FS_prompt)
    elif desc is not None:
        description = desc
    else:
        jsonfile = load_step0(task_name, challenge)
        try:
            description = get_description(jsonfile, question, img_path[0])
        except DescriptionNotFoundError as e:
            print(f"Error: {e}")
            exit(1) 
       
    # Step 1: Generate Context 
    CG_prompt = design_prompt('CG', description=description) 
    context = vlm.generate(
        instruction=[CG_prompt],
        images=img_path,
    )
    

    # Step 2: Generate Answer
    QA_prompt = design_prompt('QA', context=context, question=question)
    pred = vlm.generate(
        instruction=[QA_prompt],
        images=img_path,
    )
    if inference:
        print(f"""Description: {description}
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Context: {context}
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Prediction: {pred}"""
    ) 

    return pred

def load_step0(task_name, challenge):   
        if task_name == 'vqa':
            if challenge:
                # vqa-challenge
                jsonpath = f'./step0/{task_name}_challenge.json'
            else: 
                # vqa
                jsonpath = f'./step0/{task_name}.json'
        else:
            if challenge:
                # sketchy-challenge
                jsonpath = f'./step0/{task_name}_challenge.json'
            else:
                # sketchy
                jsonpath = f'./step0/{task_name}.json'

        jsonfile = load_json_file(jsonpath)

        return jsonfile

def get_img_path(image_base_dir_path, instance):
    return os.path.join(
        image_base_dir_path, instance
    )

def run_step0(img_base_dir, question_jsonfile, llm_name, jsonfilepath):

    '''
    
    Input:
    img_base_dir: str '../data/oodcv_images'
    question_jsonfile: str '../data/label/oodcv-vqa/oodcv-vqa.json'
    llm_name: str
    jsonfilepath: str, is path use to save jsonfile contain description generated by step0

    format:
    [{'question': Q,
    'image_path': IP, # ../data/oodcv_images/phase-2/images/659.jpg
    'Description': D
    }]

    '''
    descriptions = []

    # Run step0
    data = load_json_file(question_jsonfile)
    for index, item in tqdm(enumerate(data), total=len(data), desc='Running step0'): 
        temp = {}
        question = item['question']
        image_path =  get_img_path(img_base_dir, item['image'])
        # Step 0: Generate description by using llm
        IP_FS_prompt = design_prompt('IP-FS', question)
        description = respond(llm_name,  IP_FS_prompt)

        temp['question'] = question
        temp['image_path'] = image_path
        temp['description'] = description
        descriptions.append(temp)
        # if index == 0:
        #     print(temp)
    
    
    # Save step0
    write_to_json(jsonfilepath, descriptions)
    
    print('Run and save step0 successfully')

