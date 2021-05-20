CUDA_VISIBLE_DEVICES=0 python main.py  infer --model_type Transformer --ae_model auto_encoder.py --config config.yml --auto_config --num_gpus 1 \
    --features_file ../data/test/test.triple \
    --predictions_file eval/result \
