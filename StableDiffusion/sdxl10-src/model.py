import os
import boto3
import json
import torch
import logging
import uuid
from pathlib import Path
from djl_python import Input, Output
from diffusers import DiffusionPipeline

model_dict = None


def load_model(properties):
    s3 = boto3.client('s3')
    
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")
    
    pipe = DiffusionPipeline.from_pretrained(
        model_location,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16")
    pipe = pipe.to("cuda")
    
    model_dict = {'pipeline': pipe, 's3': s3}
    return model_dict


def handle(inputs: Input):
    global model_dict
    
    if not model_dict:
        model_dict = load_model(inputs.get_properties())
    
    pipe = model_dict['pipeline']
    s3 = model_dict['s3']
    
    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        pipe('A photo of tiger with a lion').images[0]
        return None
    
    data = inputs.get_as_json()
    input_text = data["text"]
    upload_s3_bucket = data["upload_s3_bucket"]
    with torch.inference_mode():
        with torch.autocast('cuda'):
            image = pipe(input_text).images[0]
    
    local_output = Path("/tmp/sd-output")
    local_output.mkdir(exist_ok=True)
    img_name = str(uuid.uuid4()) + ".png"
    img_path = os.path.join(local_output, img_name)
    image.save(img_path)
    
    s3_obj_path = f"stable-diffusion/sd-output/{img_name}"
    s3_full_path = f"s3://{upload_s3_bucket}/{s3_obj_path}"
    logging.info(f"Output Full path: {s3_full_path}")
    with open(img_path, "rb") as f:
        s3.upload_fileobj(f, upload_s3_bucket, s3_obj_path)
    
    output = s3_full_path
    return Output().add(output)
                