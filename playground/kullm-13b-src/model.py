import os
import json
import torch
import logging
from djl_python import Input, Output
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_dict = None


def load_model(properties):
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    
    tensor_parallel = properties["tensor_parallel_degree"]
    
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    # model = AutoModelForCausalLM.from_pretrained(model_location, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(model_location, device_map='auto', torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_8bit=True)
    model.eval()
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    model_dict = {'pipeline': pipe}
    return model_dict


def infer(pipe, params, instruction="", input_text=""):
    prompt_format = "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n"
    prompt = prompt_format.format(instruction=instruction, input=input_text)
    
    output = pipe(
        prompt,
        **params,
        # max_length=512,
        # temperature=0.7,
        # top_p=0.7,
        eos_token_id=2
    )
    s = output[0]["generated_text"]
    result = s.split("### 응답:")[1].strip()
    return result


def handle(inputs: Input):
    global model_dict
    
    if not model_dict:
        model_dict = load_model(inputs.get_properties())
    
    if inputs.is_empty():
        return None
    
    data = inputs.get_as_json()
    input_text = data["input_text"]
    instruction = data["instruction"]
    params = data["parameters"]
    pipe = model_dict["pipeline"]
    
    result = infer(pipe, params, instruction, input_text)
    return Output().add(result)
                