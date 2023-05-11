import json
from djl_python.inputs import Input
from model import handle
from deepspeed_inference import handle

# ---------- Initialization (Model loading) -------------
init_input = Input()
init_input.properties = {
    "task": "text-generation",
    "dtype": "fp16",
    "tensor_parallel_degree": 1,
    "model_dir" : "/home/ec2-user/SageMaker/efs/aiml/gen-ai-sagemaker/StableLM/pretrained-models/models--stabilityai--stablelm-base-alpha-7b/snapshots/38366357b5a45e002af2d254ff3d559444ec2147"
}
handle(init_input)


# ---------- Invocation -------------
prompt_input = Input()
prompt_input.properties = {
    "Content-Type": "application/json"
}

prompt = "How can I get some great phone?"
payload = bytes(json.dumps(
        {
            "text": [prompt],
            "inputs": [prompt]
        }
    ), 'utf-8')
prompt_input.content.add(key="data", value=payload)
output = handle(prompt_input)
print(output)


