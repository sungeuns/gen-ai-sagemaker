import os
import boto3
import json
import torch
import logging
import scipy
import uuid
from pathlib import Path
from djl_python import Input, Output
from transformers import AutoProcessor, MusicgenForConditionalGeneration

model_dict = None


def load_model(properties):
    s3 = boto3.client('s3')
    
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")
    
    processor = AutoProcessor.from_pretrained(model_location)
    model = MusicgenForConditionalGeneration.from_pretrained(model_location)
    # model.eval()
    model.to("cuda")
    
    logging.info("MusicGen model is loaded ...")
    
    model_dict = {'processor': processor, 'model': model, 's3': s3}
    return model_dict


def handle(inputs: Input):
    global model_dict
    
    if not model_dict:
        model_dict = load_model(inputs.get_properties())
    
    model = model_dict['model']
    processor = model_dict['processor']
    s3 = model_dict['s3']
    
    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    data = inputs.get_as_json()
    input_text = data["text"]
    upload_s3_bucket = data["upload_s3_bucket"]
    
    processed_inputs = processor(text=input_text, return_tensors='pt', padding=True).to("cuda")
    audio_values = model.generate(**processed_inputs, max_new_tokens=1200).cpu()
    
    local_output = Path("/tmp/musicgen-output")
    local_output.mkdir(exist_ok=True)
    audio_name = str(uuid.uuid4()) + ".wav"
    audio_path = os.path.join(local_output, audio_name)
    
    sampling_rate = model.config.audio_encoder.sampling_rate 
    scipy.io.wavfile.write(audio_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
    
    s3_obj_path = f"musicgen/musicgen-output/{audio_name}"
    s3_full_path = f"s3://{upload_s3_bucket}/{s3_obj_path}"
    logging.info(f"Output Full path: {s3_full_path}")
    
    with open(audio_path, "rb") as f:
        s3.upload_fileobj(f, upload_s3_bucket, s3_obj_path)
    
    output = s3_full_path
    return Output().add(output)
                