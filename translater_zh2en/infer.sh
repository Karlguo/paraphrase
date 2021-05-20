CUDA_VISIBLE_DEVICES=$1 python main.py  infer --model_type Transformer --ae_model model_set2seq_light.py --config config.yml --auto_config --num_gpus 1 \
    --features_file ../data/Quora/test/test.triple \
    --predictions_file ../data/Quora/test/result.back_tran \
