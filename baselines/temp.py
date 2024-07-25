'''
    pipeline:
    Step1: Main Objs = LLMs(Q) 
    Step2: Context (Caption in detail) = VLMs(Describe Main Objs) --> Location, Color, Relation, Number,...
    Step3: Prompt = {
        Context: {Context} \n
        Question: {Question} \n
        So the Answer is: .....
    }

        --> Prediction = VLMs(Prompt)
'''

from utils import design_prompt
from llms.utils import respond 


# prompt = design_prompt('IP-FS', 'huy')
# a = respond('gemma:7b', prompt)

# print(a)


from utils import load_json_file

json = load_json_file('/mnt/AI_Data/UNICORN/test/unicorn-/data/label/oodcv-vqa/oodcv-vqa.json')
print(len(json))
