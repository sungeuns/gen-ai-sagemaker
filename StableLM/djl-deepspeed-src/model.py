import os
import json
import torch
import logging
import deepspeed
from djl_python import Input, Output
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


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
    model = AutoModelForCausalLM.from_pretrained(model_location, low_cpu_mem_usage=True)
    # model.half().cuda()
    
    ds_model = deepspeed.init_inference(
        model=model,
        mp_size=tensor_parallel,
        dtype=torch.float16,
        replace_method="auto",
        replace_with_kernel_inject=True
    )
    model_dict = {'model': ds_model.module, 'tokenizer': tokenizer}
    # model_dict = {'model': model, 'tokenizer': tokenizer}
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
    
    tokenizer = model_dict['tokenizer']
    stable_lm_model = model_dict['model']
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    tokens = stable_lm_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        # stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return Output().add(output)
                