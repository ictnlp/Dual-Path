# Modeling Dual Read/Write Paths for Simultaneous Machine Translation

Source code for our ACL 2022 paper "Modeling Dual Read/Write Paths for Simultaneous Machine Translation" (PDF)

Our method is implemented based on the open-source toolkit [Fairseq](https://github.com/pytorch/fairseq).



## Requirements and Installation

- Python version = 3.6

- [PyTorch](http://pytorch.org/) version = 1.7

- Install fairseq:

  ```bash
  git clone https://github.com/ictnlp/Dual-Paths.git
  cd Dual-Paths
  pip install --editable ./
  ```

    

## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)) and WMT15 German-English (download [here](https://www.statmt.org/wmt15/)), and apply BPE with 32K merge operations on WMT15 German-English via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt).

Then, we process the data into the fairseq format:

```bash
src = SOURCE_LANGUAGE
tgt = TARGET_LANGUAGE
train_data = PATH_TO_TRAIN_DATA
vaild_data = PATH_TO_VALID_DATA
test_data = PATH_TO_TEST_DATA
data=PATH_TO_DATA

fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20
```

### Training

Train the Dual Paths SiMT with the following command:

- For IWSLT15 English-Vietnamese: we set ***latency weight*** = 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5.
- For WMT15 German-English: we set ***latency weight*** = 0.1, 0.2, 0.25, 0.3, 0.4.
- ***dual weight*** is set to 1.0 for 'Dual Paths', and set to 0.0 for 'Single Path'.

```bash
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
    --left-pad-source \
    --dual-weight 1.0 \
    --save-dir ${modelfile} \
    --max-tokens 2400 --update-freq 4
```

### Inference

Evaluate the model with the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
ref=PATH_TO_REFERENCE

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 

# generate translation
python fairseq_cli/generate.py ${data} \
    --path ${modelfile}/average-model.pt  \
    --left-pad-source \
    --batch-size 250 \
    --beam 1 \
    --remove-bpe > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref} < pred.translation
```



## Our Results

The numerical results on WMT15 German-to-English:

| **latency weight** | **AP** | **AL** | **DAL** | **BLEU** |
| :----------------: | :----: | :----: | :-----: | :------: |
|        0.4         |  0.60  |  2.80  |  4.75   |  26.21   |
|        0.3         |  0.62  |  3.19  |  5.40   |  27.04   |
|        0.25        |  0.65  |  4.02  |  6.65   |  28.14   |
|        0.2         |  0.75  |  7.69  |  11.51  |  29.23   |
|        0.1         |  0.85  | 13.50  |  17.59  |  30.10   |

More results please refer to the paper.



## Citation

In this repository is useful for you, please cite as:

```
@inproceedings{DualPaths,
	title = {Modeling Dual Read/Write Paths for Simultaneous Machine Translation},
	author = {Shaolei Zhang and Yang Feng},
	booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
	year = {2022},
}
```

