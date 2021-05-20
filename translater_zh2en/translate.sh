CUDA_VISIBLE_DEVICES=$1 python main.py  infer --model_type Transformer --config config.yml --auto_config --num_gpus 1 \
    --features_file ../data/Quora/full/data.0.bpe \
    --predictions_file ../data/Quora/full/data.0.zh \
