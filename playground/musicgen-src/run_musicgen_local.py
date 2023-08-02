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
    "model_dir": "/home/ec2-user/SageMaker/efs/aiml/gen-ai-sagemaker/playground/pretrained-models/musicgen-large"
}
handle(init_input)


# ---------- Invocation -------------
prompt_input = Input()
prompt_input.properties = {
    "Content-Type": "application/json"
}

prompt = "chillstep music when I want to listen to study"
payload = bytes(json.dumps(
        {
            "text": [prompt],
            "upload_s3_bucket": "sagemaker-us-west-2-723597067299",
        }
    ), 'utf-8')
prompt_input.content.add(key="data", value=payload)
model_output = handle(prompt_input)
print(model_output)

output = str(model_output.content.value_at(0), "utf-8")
print(output)


