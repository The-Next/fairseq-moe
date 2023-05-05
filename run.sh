NUM_EXPERTS=32
max_tokens=1024
max_updates=50000
# #DATA="/data/lsj/nfs/it_experiment/data/general_data/text_data/enfr"
DATA="/home/sxy/Projects/data_pre/data/data/opus-100-corpus/preprocessed_data/main_data_bin"
SAVE="/home/sxy/Projects/cp/moe/model/moe_model_1"
# lang_pairs="af-en,am-en,ar-en,as-en,az-en,be-en,bg-en,bn-en,br-en,bs-en,ca-en,cs-en,cy-en,da-en,de-en,el-en,eo-en,es-en,et-en,eu-en,fa-en,fi-en,fr-en,fy-en,ga-en,gd-en,gl-en,gu-en,ha-en,he-en,hi-en,hr-en,hu-en,id-en,ig-en,is-en,it-en,ja-en,ka-en,kk-en,km-en,kn-en,ko-en,ku-en,ky-en,li-en,lt-en,lv-en,mg-en,mk-en,ml-en,mr-en,ms-en,mt-en,my-en,nb-en,ne-en,nl-en,nn-en,no-en,oc-en,or-en,pa-en,pl-en,ps-en,pt-en,ro-en,ru-en,rw-en,se-en,sh-en,si-en,sk-en,sl-en,sq-en,sr-en,sv-en,ta-en,te-en,tg-en,th-en,tk-en,tr-en,tt-en,ug-en,uk-en,ur-en,uz-en,vi-en,wa-en,xh-en,yi-en,zh-en,zu-en"
# lang_dict="gl,ky,ku,mk,wa,tg,cy,eo,ml,vi,nb,lt,sl,bs,nn,li,ro,ig,am,xh,fa,pa,sk,oc,eu,is,kk,br,sv,my,sq,it,uk,mt,rw,de,ja,en,se,no,he,km,hi,ha,id,fy,el,bg,yi,ne,fi,mr,or,si,zh,ps,gd,th,da,sh,hr,fr,be,af,tr,zu,ar,as,tk,mg,ka,ru,hu,ur,nl,ga,bn,ta,tt,az,uz,es,kn,et,ug,gu,pt,te,cs,ms,ca,ko,pl,lv,sr"
lang_pairs="zh-en,de-en"
lang_dict="en,de,zh"
# export CUDA_VISIBLE_DEVICES=0,1
python train.py $DATA \
  --ddp-backend fully_sharded --fp16 --fp16-no-flatten-grads \
  --task translation_multi_simple_epoch \
  --langtoks-specs "main" \
  --langtoks "{\"main\":(\"src\", \"tgt\")}" \
  --sampling-method 'temperature' --sampling-temperature 5 \
  --langs ${lang_dict} --lang-pairs ${lang_pairs} \
  --enable-reservsed-directions-shared-datasets \
  --arch transformer \
  --share-all-embeddings \
  --encoder-normalize-before --decoder-normalize-before \
  --encoder-layers 3 --decoder-layers 3 \
  --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
  --max-source-positions 512 --max-target-positions 512 \
  --encoder-attention-heads 8 --decoder-attention-heads 8 \
  --moe-expert-count $NUM_EXPERTS --moe-freq 2 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --criterion moe_cross_entropy --moe-gate-loss-wt 0.1 --moe-gate-loss-combine-method sum \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler polynomial_decay --total-num-update $max_updates --max-update $max_updates \
  --dropout 0.2 --attention-dropout 0.2 \
  --max-tokens $max_tokens --update-freq 2 \
  --log-interval 1 \
  --save-interval-updates 3000 --save-dir $SAVE --keep-interval-updates 5 \
  --dataset-impl "mmap" \
  --record-a2a-perf-stats \
  --use-moe-pad-mask \
  --moe-batch-prioritized-routing \
  --moe-expert-output-masking 0.2 \
  --use-moe-cmr-group \
  --language-divide /home/sxy/Projects/cp/moe/json_file/test.json \
  --moe-cmr-dropout 0.2 \
  --cmr-gate-loss-p 0.95 --cmr-gate-loss-wt 0.1 \
  --enable-lang-ids 
  # --use-moe-lang-perception
#  --symlink-best-and-last-checkpoints \
# moe-eval-capacity-token-fraction # 设置为-1则在eval的时候也会使用training时候的capacity，即2*num_tokens/num_experts，这可能会导致capacity过小，丢失很多token, 使得很多token在moe layer的输出为0，即跳过moe
# --use-moe-pad-mask 不发送pad token，可能可以降低计算成本(全0行变多了)