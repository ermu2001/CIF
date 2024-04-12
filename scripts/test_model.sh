model_name=${1}
model_name_or_path=MODELS/${model_name}

PYTHONPATH=. \
python -m tasks.construct_model \
    --model_name_or_path ${model_name_or_path}
