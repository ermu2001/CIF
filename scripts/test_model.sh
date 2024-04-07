# output_dir="OUTPUTS/cifnet-18-tiny-lr0.1-attention"
seed=42

# dataset
dataset_name="cifar10"
image_column_name="img"
label_column_name="label"

# # train
# accel_config="scripts/accel_config_singlegpu.yaml --gpu_ids 0"
# output_dir="OUTPUTS/cifnet-18-tiny-lr0.01-bottleneck"
# model_name_or_path=MODELS/cifnet-18-tiny_bottleneck

accel_config="scripts/accel_config_singlegpu.yaml --gpu_ids 1"
output_dir="OUTPUTS/${model_name}"

model_name=${1}
model_name_or_path=MODELS/${model_name}

num_workers=32
learning_rate=0.01
lr_scheduler_type="cosine"
gradient_accumulation_steps=1
per_device_train_batch_size=128
# num_train_epochs=10
max_train_steps=64000
num_warmup_steps=6400
# # debug
# max_train_samples=10000

PYTHONPATH=. \
python -m tasks.construct_model \
    --model_name_or_path ${model_name_or_path}
