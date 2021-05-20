CUDA_VISIBLE_DEVICES=$1 python main.py  infer --model_type Transformer --config config.yml --auto_config --num_gpus 1 \
    --features_file ../data/Quora/small/source.bpe \
    --predictions_file ../data/Quora/small/source.zh \
