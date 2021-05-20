CUDA_VISIBLE_DEVICES=6,7 python main.py --model_type Transformer \
          --config config.yml --auto_config \
          train_and_eval --num_gpus 2
