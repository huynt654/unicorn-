from utils import load_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                    default="LlamaAdapterV2", 
                    choices=["LlamaAdapterV2", "MiniGPT4", "MiniGPT4v2",
                    "LLaVA", "mPLUGOwl", "mPLUGOwl2", "PandaGPT", "InstructBLIP2", "Flamingo", 
                    "LLaVAv1.5", "LLaVAv1.5-13B", "LLaVA_llama2", "LLaVA_llama2-13B", 
                    "MiniGPT4_llama2", "Qwen-VL-Chat", "InstructBLIP2-FlanT5-xl", 
                    "InstructBLIP2-FlanT5-xxl",  "InstructBLIP2-13B", "MiniGPT4_13B", "CogVLM", 
                    "Fuyu", "InternLM"])
    
    parser.add_argument("img_path", type=str)
    parser.add_argument("prompt", type=str)
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



if __name__ == "__main__":
    args = parse_args()
    main(args)