# python run_peft_train.py --dataset_path [LOCAL_DATASET_PATH] --epochs 3 --lr 0.0002 --per_device_train_batch_size 1 --pretrain_model_path [LOCAL_MODEL_PATH]
python run_peft_train.py --dataset_path ../samsum-data --epochs 3 --lr 0.0002 --per_device_train_batch_size 1 --pretrain_model_path ../pretrained-models/models--databricks--dolly-v2-7b/snapshots/97611f20f95e1d8c1e914b85da55cc3937c31192
