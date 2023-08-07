import json
from djl_python.inputs import Input
from model import handle

# ---------- Initialization (Model loading) -------------
init_input = Input()
init_input.properties = {
    "tensor_parallel_degree": 1,
    "model_dir": "/home/ec2-user/SageMaker/efs/aiml/gen-ai-sagemaker/playground/pretrained-models/llama2-chat/13b"
}
handle(init_input)


# ---------- Invocation -------------
prompt_input = Input()
prompt_input.properties = {
    "Content-Type": "application/json"
}

prompt = "Today is very stressful day. How can I make my feel better?"
instruction = """
You are a friendly and knowledgeable assistant named SESO.
Your should introduce yourself first.
Be comforting, empathetic, and make them feel as good as possible about their questions.
"""

# payload = bytes(json.dumps(
#         {
#             "text": [prompt],
#             "instruction": [instruction]
#         }
#     ), 'utf-8')

payload = bytes(json.dumps(
        {
            "text": prompt,
            "instruction": instruction,
            "parameters": {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.6,
                "repetition_penalty": 1.03,
            }
        }
    ), 'utf-8')
prompt_input.content.add(key="data", value=payload)
model_output = handle(prompt_input)
print(model_output)

output = str(model_output.content.value_at(0), "utf-8")
print(output)


