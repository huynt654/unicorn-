from utils import load_model, design_prompt
import argparse
from llms.utils import response

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
    # "How many unicorns would there be in the image after four more unicorns have been added in the image?"
    # "How many unicorns would there be in the image after four more unicorns have been added in the image? \n The main objects mentioned in the question are: Unicorns. \n Instruction:  Describe the surrounding objects and characteristics of the main object (if any). \n Answer the questions in a thorough analytical way.")

    parser.add_argument("--img_path", type=str, default='/mnt/AI_Data/UNICORN/test/demo.jpg')
    parser.add_argument("--question", type=str, default="How many cats in this image?")

    args = parser.parse_args()
    return args

def main(args):

    # init vlm model
    vlm_model = load_model(args.model_name)

    # Step 0: Generate description 
    IP_FS_prompt = design_prompt('IP-FS', args.question)
    print(IP_FS_prompt)
    description_instruction = response(args.llm_name,  IP_FS_prompt)
    print(description_instruction)
    # Step 1: Generate Context
    CG_prompt = design_prompt(description_instruction)
    print(CG_prompt)
    context = vlm_model.generate(
        instruction=[CG_prompt],
        images=[args.img_path],
    )

    # Step 2: Generate Answer
    QA_prompt = design_prompt('QA', context=context, question=args.question)
    pred = vlm_model.generate(
        instruction=[QA_prompt],
        images=[args.img_path],
    )

    print(f'Instruction:\t{QA_prompt}')
    print(f'Answer:\t{pred}')

if __name__ == "__main__":
    args = parse_args()
    main(args)






''' a = {
• Màu sắc
• Kích thước
• Hình dạng
• Vị trí trong hình
• Số lượng (nếu có nhiều object tương tự)
• Chất liệu hoặc kết cấu
• Mối tương quan với các object khác
• Chức năng hoặc mục đích sử dụng (nếu có thể nhận biết)
• Tình trạng (mới, cũ, hư hỏng, v.v.)
• Chi tiết đặc biệt hoặc nổi bật
• Góc nhìn hoặc hướng của object
• Ánh sáng và bóng đổ
• Biểu cảm hoặc tư thế (nếu là sinh vật)
• Phong cách hoặc thời đại (nếu áp dụng)
• Bối cảnh xung quanh object
}
'''

