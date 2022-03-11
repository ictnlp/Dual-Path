export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
lambda=LATENCY_WEIGHT

python train.py  --ddp-backend=no_c10d ${data} --arch transformer_monotonic_iwslt_de_en --share-all-embeddings \
    --user-dir ./examples/simultaneous_translation \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --max-update 180000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --dual-weight 1.0 \
    --left-pad-source \
    --save-dir ${modelfile} \
    --max-tokens 2400 --update-freq 4