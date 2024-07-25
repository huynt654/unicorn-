from utils import load_model, design_prompt
import argparse
from llms.utils import respond
import torch
'''
    pipeline:
    Step0: Description = LLMs(Q) 
    Step1: Context (Caption in detail) = VLMs(Description) 
    Step2: Prompt = {
        Context: {Context} \n
        Question: {Question} \n
        So the Answer is: .....
    }
        --> Prediction = VLMs(Prompt)
'''

def parse_args():

    # pwd == baselines
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
   
    parser.add_argument("--llm_name", type=str, 
                        default='gemma:7b',
                        choices=[
                            'gemma:7b',
                            'gemma:2b',
                            'llama3:8b',
                            'qwen2:7b'
                        ])
    
    parser.add_argument("--img_path", type=str, default='/mnt/AI_Data/UNICORN/test/demo3.png')
    parser.add_argument("--question", type=str, default="What does the man who sits have trouble doing? \n Multiple Choice Answers: \n (a) Riding (b) Breathing (c) Walking (d) Magic")

    args = parser.parse_args()
    return args

def main(args):

    # init vlm model
    vlm_model = load_model(args.model_name)

    # Step 0: Generate description llm
    IP_FS_prompt = design_prompt('IP-FS', args.question)
    description_instruction = respond(args.llm_name,  IP_FS_prompt)
    print(description_instruction)

    # description = "A description of the man's physical condition or any visible signs of difficulty performing the listed activities."
    # # Step 1: Generate Context 
    # CG_prompt = design_prompt('CG', description=description) #_instruction)
    # context = vlm_model.generate(
    #     instruction=[CG_prompt],
    #     images=[args.img_path],
    # )
    # print(context)



    # context = 'The man is described as elderly and using a wheelchair, which suggests that he may have some physical limitations or difficulty walking. However, the image does not provide enough information to determine the severity of his physical condition or any visible signs of difficulty performing the listed activities.'
    # # Step 2: Generate Answer
    # QA_prompt = design_prompt('QA', context=context, question=args.question)
    # pred = vlm_model.generate(
    #     instruction=[QA_prompt],
    #     images=[args.img_path],
    # )
    # print(f'Instruction:\t{QA_prompt}')
    # print(f'Answer:\t{pred}')


if __name__ == "__main__":
    args = parse_args()
    main(args)


