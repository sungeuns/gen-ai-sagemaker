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
    # "model_dir" : "/home/ec2-user/efs/seso/code/aiml/gen-ai-aws/gen-ai-app/ml_src/pretrained-models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/47cd5302d866fa60cf8fb81f0e34d42e38f6100c",
    "model_dir": "/home/ec2-user/SageMaker/efs/aiml/gen-ai-sagemaker/StableDiffusion/pretrained-models/sdxl10-base"
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


