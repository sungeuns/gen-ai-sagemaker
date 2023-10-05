import os
import json
import torch
import logging
from djl_python import Input, Output
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


model_dict = None


def load_model(properties):
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")
    tokenizer = AutoTokenizer.from_pretrained(model_location, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_location, device_map="auto", torch_dtype=torch.bfloat16)
    
    generate_text = InstructionTextGenerationPipeline(
        model = model,
        tokenizer = tokenizer,
        do_sample = True,
        max_new_tokens = 128
    )
    
    model_dict = {'model': model, 'tokenizer': tokenizer, 'generate_text': generate_text}
    return model_dict


def handle(inputs: Input):
    global model_dict
    
    if not model_dict:
        model_dict = load_model(inputs.get_properties())
    
    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    data = inputs.get_as_json()
    input_text = data["text"]
    
    generate_text = model_dict['generate_text']
    output = generate_text(input_text)
    return Output().add(output)
                