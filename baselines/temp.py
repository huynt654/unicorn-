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


from utils import load_json_file, run_step0, get_description, load_step0, load_model, pipeline

json = load_json_file('/mnt/AI_Data/UNICORN/test/unicorn-/data/label/oodcv-vqa/oodcv-vqa.json')
print(len(json))

# run step0
# pipeline 

# run_step0(img_base_dir='../data/oodcv_images', question_jsonfile='../data/label/oodcv-vqa/oodcv-vqa.json', llm_name='gemma:7b', jsonfilepath='./step0/vqa.json')
# jsonfile = load_step0('vqa', 0)
# desc = get_description(jsonfile, "Is there a car in the image?", "../data/oodcv_images/phase-1/images/car_+_pokemon_google_0013.jpg")
# print(desc)

vlm = load_model('QwenChat')
pred = pipeline(vlm, 'gemma:7b', ["../data/oodcv_images/phase-1/images/car_+_snow_OR_snowy_google_0019.jpg"], "How many cars are there in the image?", 'vqa', 0, True, desc="Let's describe the exact number of cars in the image (write in digit).")


'''
- dùng thử llm sinh ra các task demonstration khác (tức là, input: question và đoạn description, rồi hỏi llm sinh task demonstration sao để đối với question, làm sao để sinh ra được đoạn description đó)
- custom lại prompt (IP-FS)
- chạy lại description.json 
- Kĩ thuật sinh task demonstration (idea)
'''

# ood-vqa: how many (number), is there (resemble or is) và 1 câu add/remove 

# challenge-vqa: (no change(remove/add zero), add/remove)
# How many unicorns would there be in the image after no unicorn was removed in the image? (add 0) (digit)
# Would there be a bus in the image after the bus disappeared from this picture? (remove 1) (yes no)
# digit (no change, change), yes/no (would there)

# --> xác định ra 3 Q 
# --> nghĩ ra 3 description 
# --> dùng llm và description để sinh ra task demonstration 
# --> dùng llm và task demonstration để sinh ra các task demonstration khác cùng semantic 
# --> chọn 1 task demonstration tối ưu
# --> viết lại design prompt
# sinh lại description.json
# run eval


'''
sketchy (yes/no question) --> chỉ khác label, ở challenge có label khó nhận diện hơn 
Is this a/an {} in the image?
In the scene, is a/an {} in it? 
Is there a sketchy {} in the picture? 
'''

'''
Let's describe objects that resemble or are buses (describe in one setence)

Let's describe the number of bicycles in the picture (describe in one setence)

"Is there a bus in the image?"  Hãy mô tả về những object giống hoặc là bus
"How many bicycles are there in the image?" Hãy mô tả số lượng bicycle có trong hình
"How many unicorns would there be in the image after no unicorn was removed in the image?"
"Would there be a bicycle in the image after the bicycle disappeared from this picture?"
'''

'''
- table1: accuracy over 6 datasets
- ablation study:
+ test --> describe in one, 2, 3,... sentence
+ test --> task desmontrastion biến thiên 
- quantitative experirenmetal: example some sample
- With and without: description + context 
'''




'''
Your suggestion for a task demonstration is a good starting point. To make it more specific and aligned with the examples provided, we can refine it slightly. Here's an optimal task demonstration in the form of a question:

Task demonstration: What specific aspect of the image should we focus on describing to directly address the question above?

This task demonstration works well because:

1. It prompts the AI to consider the specific elements of the image that are relevant to the question.
2. It encourages a focused response that aligns with the question's intent.
3. It can be applied to a wide range of image-related questions, similar to the examples provided.
4. It naturally leads to instructions that guide the description process, as seen in the examples.

This task demonstration can effectively generate instructions like those in the examples:

- For "Is there a bus in the image?", it would lead to an instruction like "Let's describe objects that resemble or are buses."
- For "How many bicycles are there in the image?", it would result in an instruction like "Let's describe the number of bicycles in the picture."

This format allows for flexibility while maintaining a clear focus on the specific information needed to answer the original question.
'''

# Example 1:
# Question: "Is there a bus in the image?" 
# Task demonstration: What specific aspect of the image should we focus on describing to directly address the question above? (describe in one sentence)
# Answer: 'Let's describe objects that resemble or are buses. (if there are buses)'

# Example 2: 
# Question: "How many cars are there in the image?"
# Task demonstration: What specific aspect of the image should we focus on describing to directly address the question above? (describe in one sentence)
# Answer: 'Let's describe the number of objects that resemble or are cars in the image (if there are buses).'

# Example 3:
# Question: "How many bicycles are there in the image?"
# Task demonstration: What specific aspect of the image should we focus on describing to directly address the question above? (describe in one sentence)
# Answer: 'Let's observe carefully under many conditions (view, weather,...) and write to describe the number of bycicles in the image (if there are bycicles).'

# Question: "How many chairs are there in the image?"
# Task demonstration: What specific aspect of the image should we focus on describing to directly address the question above? (describe in one sentence)
# Answer:   