from utils import load_model
import argparse

# configs = json.load(open("./config.json"))
# DATA_DIR = configs['DATA_DIR']



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


model = load_model("InternLM")



