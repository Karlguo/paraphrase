GPU=$1
DATA=$2
OUTPUT=$3

if [[ ! -n "$DATA" || ! -f "$DATA" ]]; then
    echo "Cannot find Data file $DATA, try again"
    exit
fi

if [ ! -n "$OUTPUT" ]; then
    echo "No output file, using default output default_make_data/"
    OUTPUT=default_make_data/
fi

rm -rf tmp/split_data
rm -rf $OUTPUT/result
rm -rf $OUTPUT/log

mkdir -p tmp/split_data
mkdir -p $OUTPUT/result
mkdir -p $OUTPUT/log

gpu_list=(${GPU//,/ })
num_gpu=${#gpu_list[@]}

count=`cat $DATA | wc -l`

line=$[count / num_gpu]
if [ $[count % num_gpu] != 0 ]; then
    line=$[line + 1]
fi

echo "== Spliting data into $num_gpu parts"
split -l $line $DATA -d -a 2 tmp/split_data/data_

pre_gpu=0
for f in `ls tmp/split_data`
do
    echo "== Using GPU ${gpu_list[$pre_gpu]} training tmp/slite_data/$f"
    echo "== Log in $OUTPUT/log"
    echo "== Result in $OUTPUT/result"

    CUDA_VISIBLE_DEVICES=${gpu_list[$pre_gpu]} nohup python main.py infer \
        --model_type Transformer \
        --ae_model model_set2seq.py \
        --config config.yml \
        --auto_config \
        --num_gpus 1\
        --features_file tmp/split_data/$f \
        --predictions_file $OUTPUT/result/result.$f \
        > $OUTPUT/log/log.${f} &

    pre_gpu=$[pre_gpu + 1]
done

#CUDA_VISIBLE_DEVICES=$GPU python translate.py \
#  --src_vocab_file vocab_en2zh/src.vocab.txt \
#  --trg_vocab_file vocab_en2zh/trg.vocab.txt \
#  --file data/split_zh2en/en/$DATA \
#  --file_trg data/split_zh2en/zh/$DATA \
#  --file_out data/result/$DATA \
#  --config_file config.en2zh.yaml \
