CUDA_VISIBLE_DEVICES=3,2 python main.py --model model_set2seq_light.py \
          --config config.yml --auto_config \
          train_and_eval --num_gpus 2
