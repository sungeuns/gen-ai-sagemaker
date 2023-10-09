
## Fine-tuning LLM using Dolly

### 로컬 모드 환경
- LLM 모델을 local 에서 inference 해 보거나, fine-tuning 등을 테스트해 보기 위해서는 local storage가 `30GB` 이상 필요합니다. (`40~50GB 이상`을 추천합니다)
- 7B 크기의 LLM 모델의 경우 `g4dn.2xlarge` 이상의 인스턴스를 추천하며, 안정적으로 사용하기 위해서는 `g5.2xlarge` 이상의 인스턴스를 사용하는 것이 좋습니다. (16bit 기준, 8bit quantization 시 g4dn으로도 안정적으로 가능)

### 실습
- `01-local-mode-dolly-inference.ipynb`
  - Local mode로 LLM을 테스트 해봅니다.
- `02-sagemaker-endpoint-dolly.ipynb`
  - DJL를 활용하여 LLM을 SageMaker endpoint로 배포합니다.
- `03-supervised-fine-tuning-peft.ipynb`
  - PEFT를 활용하여 SageMaker 상에서 LLM을 Supervised fine tuning 합니다.
- `04-fine-tuning-rlhf.ipynb`
  - TRL, PEFT 등을 활용하여 SageMaker 상에서 LLM을 RLHF 기반 학습을 진행합니다.
- `05-stack-dolly.ipynb`
  - [StackLLaMA](https://huggingface.co/blog/stackllama)를 SageMaker에서 학습하는 예시로, SFT에 대한 샘플 코드만 포함되어 있어서 이를 참고하여 추가적인 학습을 직접 진행해 보도록 합니다.
  

### SageMaker notebook lifecycle configuration

- `lifecycle_configuration` 디렉토리 내부의 파일을 참고하여 설정하면 SageMaker notebook instance에서 VSCode를 활용할 수 있습니다.
- VSCode를 활용하면 로컬 모드로 디버깅을 훨씬 수월하게 할 수 있습니다.

