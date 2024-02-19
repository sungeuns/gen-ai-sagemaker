import os
import json
import torch
import logging
from djl_python import Input, Output
import random
import boto3
import uuid
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video


model_dict = None


def divide_bucket_key_and_filename_from_s3_uri(s3_uri):
    # Parse the S3 URI
    parsed_uri = urlparse(s3_uri)

    # Extract the bucket name
    bucket = parsed_uri.netloc

    # Extract the key (object path)
    key = parsed_uri.path.lstrip("/")

    # Extract the filename
    filename = key.split("/")[-1]
    return bucket, key, filename


def load_model(properties):
    with_unet_compile = False
    
    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")
    
    tensor_parallel = properties["tensor_parallel_degree"]
    
    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_location, torch_dtype=torch.float16, variant="fp16"
    )

    if with_unet_compile:
        # Need pytorch 2.2
        # refer to https://github.com/huggingface/diffusers/issues/6096
        pipe.to("cuda")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    else:
        pipe.enable_model_cpu_offload()
    
    s3 = boto3.client('s3')
    model_dict = {'pipeline': pipe, "s3": s3}
    return model_dict


def infer(pipe, image, params):
    image = image.resize((1024, 576))
    generator = torch.manual_seed(random.randint(0, 999999))
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

    output_dir = Path("/tmp/video-output")
    output_dir.mkdir(exist_ok=True)
    vid_name = str(uuid.uuid4()) + ".mp4"
    logging.info(f"Output video name: {vid_name}")
    ouptut_file_path = os.path.join(output_dir, vid_name)
    export_to_video(frames, ouptut_file_path, fps=7)
    logging.info(f"Output video local path: {ouptut_file_path}")

    return ouptut_file_path, vid_name


def handle(inputs: Input):
    global model_dict
    
    if not model_dict:
        model_dict = load_model(inputs.get_properties())
    
    if inputs.is_empty():
        return None
    
    pipe = model_dict["pipeline"]
    s3_client = model_dict['s3']
    
    data = inputs.get_as_json()
    input_image_s3 = data["input_image_s3"]
    upload_s3_bucket = data["upload_s3_bucket"]
    params = data["parameters"]
    
    bucket, key, fname = divide_bucket_key_and_filename_from_s3_uri(input_image_s3)
    local_dir = Path("/tmp/image-input")
    local_dir.mkdir(exist_ok=True)
    local_file_path = os.path.join(local_dir, fname)
    s3_client.download_file(Bucket=bucket, Key=key, Filename=local_file_path)
    logging.info(f"Local image path: {local_file_path}")
    raw_image = Image.open(local_file_path).convert('RGB')
    output_path, output_filename = infer(pipe, raw_image, params)

    # upload to s3
    s3_obj_path = f"svd-xl/svd-xl-output/{output_filename}"
    s3_full_path = f"s3://{upload_s3_bucket}/{s3_obj_path}"
    with open(output_path, "rb") as f:
        s3_client.upload_fileobj(f, upload_s3_bucket, s3_obj_path)
        
    logging.info(f"Output video upload to S3: {s3_full_path}")
    
    result = {
        "s3_path" : s3_full_path
    }
    
    return Output().add(result)
                