## Generative AI on SageMaker


### StableLM

It explains how to test and deploy StableLM on SageMaker

1) Test StableLM in local mode
2) Deploy StableLM model on SageMaker enpodint using DJL
3) How to create StableVicuna 13B model
  - Note that you need LLaMA model first


### StableDiffusion

It explains how to test and deploy StableDiffusion on SageMaker

1) Test StableDiffusion in local mode
2) Deploy StableDiffusion model on SageMaker enpodint


### Dolly

It explains how to test, deploy, and fine-tune (SFT, RLHF) Dolly on SageMaker
- Check the `Dolly` directory.
- The example is explained using Korean.


### Package version

- All examples are tested on Python 3.9. Local mode notebook have a version information of python packages.
- If it has error when deploy to endpoint, check the local mode version and match with this version in the `requirements.txt`


### How to local debug DJL

Need to use conda env in Sagemaker notebook

```
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv
conda env list
conda activate pytorch_p310
```

Install DJL in local env
  - https://github.com/deepjavalibrary/djl-serving/tree/master/engines/python
  

```
git clone https://github.com/deepjavalibrary/djl-serving.git
cd djl-serving
cd engines/python/setup
pip install -U -e .
```

