
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
