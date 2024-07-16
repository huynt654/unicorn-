from utils import load_model, write_to_json, load_img_paths, design_prompt
import argparse


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
    
    parser.add_argument("img_dir", type=str, default='../data/oodcv_images')
    parser.add_argument("questions", type=str, default=)
    parser.add_argument("jsonfile", type=str)
    
    args = parser.parse_args()
    return args


def main(args):

    # init model
    model = load_model(args.model_name)

    img_paths = load_img_paths(args.img_dir)

    design_prompt

    preds = []

    for img_path in img_paths:

        # prediction
        pred = model.generate(
            instruction=[args.prompt],
            images=[img_path],
        )
        prediction = {
            'prediction': pred,
            'image_path': img_path
        }
        preds.append(prediction)


    # eval 
    write_to_json(args.jsonfile, preds)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)