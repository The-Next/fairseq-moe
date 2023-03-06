
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 
models="/data/lsj/nfs/moe/moe_model3"
DATA="/data/lsj/nfs/moe/moe_data"
src="de"
tgt="en"
max_tokens=4096
resdir="/data/lsj/nfs/moe/moe_res"
lang_dict="en,de" #,es,fi,hi,ru,zh"

python -m torch.distributed.launch --nproc_per_node=6 --master_addr="127.0.0.1" --master_port=12345 \
generate.py $DATA \
--task translation_multi_simple_epoch \
-s "$src" -t "$tgt" \
--lang-pairs "$src-$tgt" --langs "$lang_dict" \
--is-moe --path "$models/checkpoint_last.pt" --fp16 \
--max-tokens "$max_tokens" --beam 3 --sacrebleu \
--remove-bpe --results-path $resdir \
--langtoks-specs "main" \
--langtoks "{\"main\":(\"src\", \"tgt\")}"

