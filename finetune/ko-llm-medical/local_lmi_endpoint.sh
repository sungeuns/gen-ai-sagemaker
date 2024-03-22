# ECR login
# aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# Get deepspeed image
# docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.25.0-deepspeed0.11.0-cu118

# test docker locally
#docker run -it 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.25.0-deepspeed0.11.0-cu118 /bin/bash

MODEL_REPO_DIR=$PWD/ko-llm-med-src
MODEL_LOG_DIR=$PWD/local-logs
mkdir -p $MODEL_LOG_DIR

docker run -it --runtime=nvidia --gpus all --shm-size 12g \
 -v $MODEL_REPO_DIR:/opt/ml/model:ro \
 -v $MODEL_LOG_DIR:/opt/djl/logs \
 -p 8080:8080 \
 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.25.0-deepspeed0.11.0-cu118

 
 