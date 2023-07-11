from djl_python import Input, Output
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Tuple
import warnings
import deepspeed


predictor = None


def get_model(properties):
    model_name = properties["model_id"]
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    tensor_parallel = properties["tensor_parallel_degree"]
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    model = deepspeed.init_inference(model, mp_size=tensor_parallel)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    generator = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
    
    return generator


def handle(inputs: Input) -> None:
    global predictor
    
    if not predictor:
        predictor = get_model(inputs.get_properties())
    
    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    data = inputs.get_as_json()
    text = data["text"]
    text_length = data["text_length"]
    
    outputs = predictor(text, do_sample=True, min_length=text_length, max_length=text_length)
    
    result = {"outputs": outputs}
    
    return Output().add_as_json(result)