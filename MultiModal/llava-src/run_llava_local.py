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
    "model_dir": "../pretrained-models/models--liuhaotian--llava-v1.5-13b/snapshots/d64eb781be6876a5facc160ab1899281f59ef684"
}
handle(init_input)


# ---------- Invocation -------------
prompt_input = Input()
prompt_input.properties = {
    "Content-Type": "application/json"
}

prompt = "What is the value in the first row of table?"
payload = bytes(json.dumps(
        {
            "text": [prompt],
            "input_image_s3": "s3://sagemaker-us-west-2-723597067299/llm/llava/input-samples/test_01.jpg",
        }
    ), 'utf-8')
prompt_input.content.add(key="data", value=payload)
model_output = handle(prompt_input)
print(model_output)

output = str(model_output.content.value_at(0), "utf-8")
print(output)


