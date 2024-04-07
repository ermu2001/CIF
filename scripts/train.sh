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

model_name=cifnet-18-tiny_basic
exp_name=default
model_name_or_path=MODELS/${model_name}

num_workers=32
learning_rate=0.001
lr_scheduler_type="cosine"
gradient_accumulation_steps=1
accel_config="scripts/accel_config_multiplegpu.yaml --gpu_ids 0,1"
# per_device_train_batch_size=128
per_device_train_batch_size=64 # ensure global batch size is 128

# num_train_epochs=10
max_train_steps=64000
num_warmup_steps=6400
# # debug
# max_train_samples=10000
output_dir="OUTPUTS/${model_name}--lr${learning_rate}--${exp_name}"




mkdir -p ${output_dir}
echo "TESTING MODEL"
PYTHONPATH=. \
python -m tasks.construct_model \
    --model_name_or_path ${model_name_or_path} > ${output_dir}/test_model.log


PYTHONPATH=. \
accelerate launch --config_file ${accel_config} tasks/train.py \
    --output_dir ${output_dir} \
    --seed ${seed} \
    --model_name_or_path ${model_name_or_path} \
    --num_workers ${num_workers} \
    --learning_rate ${learning_rate} \
    --dataset_name ${dataset_name} \
    --image_column_name ${image_column_name} \
    --label_column_name ${label_column_name} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --max_train_steps ${max_train_steps} \
    --num_warmup_steps ${num_warmup_steps} \
    --report_to tensorboard \
    --with_tracking