
## Fine-tuning LLM using Dolly

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

