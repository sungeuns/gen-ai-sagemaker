import os
import json
import torch
import logging
# import deepspeed
from djl_python import Input, Output
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

model_dict = None
stop_words = ["</s>"]
stopping_criteria = None


def load_model(properties):
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    
    tensor_parallel = properties["tensor_parallel_degree"]
    
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModelForCausalLM.from_pretrained(model_location, device_map='auto', torch_dtype=torch.float16, load_in_8bit=True)
    
    # # Deepspeed init - Not using DS, since it's strange
    # ds_model = deepspeed.init_inference(
    #     model=model,
    #     mp_size=tensor_parallel,
    #     dtype=torch.float16,
    #     replace_method="auto",
    #     replace_with_kernel_inject=True
    # )
    # model_dict = {'model': ds_model.module, 'tokenizer': tokenizer}
    
    model_dict = {'model': model, 'tokenizer': tokenizer}
    return model_dict

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message.content}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt


def handle(inputs: Input):
    global model_dict
    global stop_words
    global stopping_criteria
    
    if not model_dict:
        model_dict = load_model(inputs.get_properties())
        tokenizer = model_dict['tokenizer']
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    data = inputs.get_as_json()
    user_query = data["text"]
    system_prompt = data["instruction"]
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_query})
    prompt = build_llama2_prompt(messages)
    
    tokenizer = model_dict['tokenizer']
    llm_model = model_dict['model']    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]    
    params = data.get("parameters")
    output_tokens = llm_model.generate(
        **inputs,
        **params,
        stopping_criteria=stopping_criteria
    )
    
    output = tokenizer.decode(output_tokens[0][input_length:]).replace("</s>", "")
    
    # token = tokens.sequences[0, input_length:]
    # output_str = tokenizer.decode(token)
    # output = remove_stopword(output_str, stop_words)
    
    # output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return Output().add(output)
                