

## PEFT를 활용한 LLM fine-tuning

한국어를 지원하는 LLM (base model) 을 특정한 데이터에 맞추어서 fine-tuning 하는 예시입니다.
- PEFT를 활용하여 QLoRA 알고리즘으로 작은 GPU instance에서 학습을 진행 해 봅니다.
- 여기서는 medical QA 데이터를 활용합니다. 일반적인 케이스는 아니지만 base model 과 fine-tuning 모델의 차이가 크게 나는 것을 확인해 보기 위해서 특수한 데이터를 사용하였습니다.
- 해당 예시에서는 Upstage가 만든 Solar 기반의 모델을 활용하였습니다.


### 실습 진행

- 01_local_inference_testing.ipynb
  - base 모델을 받아서 기본적인 inference를 진행 해 봅니다.
- 02_prepare_dataset.ipynb
  - 데이터셋을 받아서 학습 가능한 형태로 정제합니다.
- 03_local_train_debugging.ipynb
  - Jupyter notebook에서 학습을 진행해 봅니다.
- 04_sagemaker_training.ipynb
  - SageMaker managed training 기능을 활용해서 학습을 진행해 봅니다.
- 05_sagemaker_lmi_dlc_endpoint.ipynb
  - Fine-tuning 된 모델을 LMI DLC를 활용해 SageMaker endpoint로 배포해 봅니다.


### 테스트 된 환경

- 로컬 테스트를 위한 Jupyter notebook의 경우 SageMaker notebook 으로 테스트 되었습니다. (`g4dn.xlarge` 이상 추천, 로컬 학습까지 하는 경우 `2xlarge` 이상)
- Fine-tuning은 `g5.4xlarge` 이상의 인스턴스를 활용하는 것을 추천하며 inference의 경우 quantization을 하는 경우 `g4dn.xlarge` 이상을 추천합니다.

```
peft==0.9.0
accelerate==0.28.0
transformers==4.38.2
bitsandbytes==0.43.0
datasets==2.18.0
```

### 참고

코드 및 관련 정보 참고
- https://github.com/daekeun-ml/genai-ko-LLM/tree/main/fine-tuning

샘플 데이터 참고
- https://huggingface.co/datasets/sean0042/KorMedMCQA

Base model 참고
- https://huggingface.co/LDCC/LDCC-SOLAR-10.7B (v1.2)

