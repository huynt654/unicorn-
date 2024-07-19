from utils import load_model
import argparse

# configs = json.load(open("./config.json"))
# DATA_DIR = configs['DATA_DIR']

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
    # "How many unicorns would there be in the image after four more unicorns have been added in the image?"
    # "How many unicorns would there be in the image after four more unicorns have been added in the image? \n The main objects mentioned in the question are: Unicorns. \n Instruction:  Describe the surrounding objects and characteristics of the main object (if any). \n Answer the questions in a thorough analytical way.")

    parser.add_argument("--img_path", type=str, default='../data/oodcv_images/phase-2/images/223.jpg')
    parser.add_argument("--prompt", type=str, default="How many unicorns would there be in the image after four more unicorns have been added in the image? Let's think step by step")

    args = parser.parse_args()
    return args

def main(args):
    
    # init model
    model = load_model(args.model_name)

    # prediction
    pred = model.generate(
        instruction=[args.prompt],
        images=[args.img_path],
    )

    print(f'Instruction:\t{args.prompt}')
    print(f'Answer:\t{pred}')

    '''
    pipeline:
    Step1: Main Objs = LLMs(Q) 
    Step2: Context (Caption in detail) = VLMs(Describe Main Objs)
    Step3: Prompt = {
        Context: {Context} \n
        Question: {Question} \n
        So the Answer is: .....
    }

        --> Prediction = VLMs(Prompt)
    '''

if __name__ == "__main__":
    args = parse_args()
    main(args)






