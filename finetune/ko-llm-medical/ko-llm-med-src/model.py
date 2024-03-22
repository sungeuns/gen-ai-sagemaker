import os
import json
import torch
import logging
from djl_python import Input, Output
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


model_dict = None


def load_model(properties):
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    
    tensor_parallel = properties["tensor_parallel_degree"]
    
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")
    
    data_type = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16
    
    quant_4bit = True
    quant_8bit = False

    if quant_4bit:
        nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=data_type
        )
    else:
        nf4_config = None
        
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_location,
        device_map='auto',
        load_in_8bit=True if quant_8bit else False,
        torch_dtype=data_type,
        quantization_config=nf4_config,
    )
    
    model_dict = {'model': model, 'tokenizer': tokenizer}
    return model_dict


def infer(llm_model, llm_tokenizer, params, instruction="", input_text=""):

    text = f"""
<s> ### User:
{instruction}

{input_text}

### Assistant:
"""
    
    model_inputs = llm_tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        **params
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


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
    llm_model = model_dict["model"]
    llm_tokenizer = model_dict["tokenizer"]
    
    result = infer(llm_model, llm_tokenizer, params, instruction, input_text)
    
    return Output().add(result)
                