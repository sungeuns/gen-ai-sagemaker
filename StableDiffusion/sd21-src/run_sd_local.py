import json
from djl_python.inputs import Input
# from stable_diffusion import handle
# from model_ds import handle
# from model_nods import handle
from model import handle

# ---------- Initialization (Model loading) -------------
init_input = Input()
init_input.properties = {
    "tensor_parallel_degree": 1,
    # "model_dir" : "/home/ec2-user/SageMaker/efs/aiml/StableDiffusion/pretrained-models/models--stabilityai--stable-diffusion-2/snapshots/07753ec23aeaf08862a5f6a8fcb0f9a883863b1b"
    "model_dir" : "/home/ec2-user/SageMaker/efs/aiml/StableDiffusion/pretrained-models/models--stabilityai--stable-diffusion-2-1/snapshots/36a01dc742066de2e8c91e7cf0b8f6b53ef53da1"
}
handle(init_input)


# ---------- Invocation -------------
prompt_input = Input()
prompt_input.properties = {
    "Content-Type": "application/json"
}

prompt = "John snow from game of throne, disney style"
payload = bytes(json.dumps(
        {
            "text": [prompt],
            "upload_s3_bucket": "sagemaker-us-west-2-723597067299",
            "prompt": prompt
        }
    ), 'utf-8')
prompt_input.content.add(key="data", value=payload)
output = handle(prompt_input)
print(output)


