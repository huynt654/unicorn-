import ollama

# document: https://github.com/huynt654/ollama-python

# Note: You must run command (ollama pull gemma:7b) in terminal before calling model to pull model

def response(llm_name, prompt):

    response = ollama.generate(model=llm_name, 
                                     prompt=prompt)

    return response['response']

def chat(llm_name, message):

    '''
    llm_name: str
    message: list[dict] 
    Example:
    [{
        'role': 'user',
        'content': 'Why is the sky blue?',
    }]
    '''

    response = ollama.chat(model=llm_name, 
                                messages=message)


    return response['message']['content']

def lister():
    supported_model_list = ollama.list()

    return supported_model_list

def show(llm_name):
    temp = ollama.show(llm_name)
    return temp

def streaming(llm_name, message):

    '''
    llm_name: str
    message: list[dict] 
    Example:
    [{'role': 'user', 'content': 'Why is the sky blue?'}]
    '''

    stream = ollama.chat(
        
        model=llm_name,
        messages=message,
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def embedding(llm_name, prompt):

    '''
    llm_name: str
    prompt: str
    Example:
    'The sky is blue because of rayleigh scattering'

    return 
    list: embedding vector
    '''

    embedding = ollama.embeddings(model=llm_name, prompt=prompt)

    return embedding['embedding']

