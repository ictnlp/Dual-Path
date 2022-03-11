#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

import pdb


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    # pdb.set_trace()

    tmp = src_dict
    src_dict = tgt_dict
    tgt_dict = tmp

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    rws = []
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        # pdb.set_trace()

        def make_back_sample(sample):
            back_sample = {}
            back_sample["id"] = sample["id"].contiguous()
            back_sample["nsentences"] = sample["nsentences"]
            back_sample["ntokens"] = (sample["net_input"]["src_tokens"] > 1).sum()
            back_sample["target"] = sample["net_input"]["src_tokens"].contiguous()

            def produce_groundtruth_back_data(
                src_tokens, src_lengths, prev_output_tokens
            ):
                back_data = {}
                bsz = src_tokens.size(0)
                src_padding_num = (prev_output_tokens == 1).sum(dim=1)
                tgt_padding_num = (src_tokens == 1).sum(dim=1)
                src_tmp = (
                    torch.arange(0, prev_output_tokens.size(1), device="cuda")
                    .unsqueeze(0)
                    .repeat(bsz, 1)
                )
                src_tmp = (
                    src_tmp
                    == (prev_output_tokens.size(1) - src_padding_num - 1).unsqueeze(1)
                ).type_as(src_tokens)

                tgt_tmp = (
                    torch.arange(0, src_tokens.size(1), device="cuda")
                    .unsqueeze(0)
                    .repeat(bsz, 1)
                )
                tgt_tmp = (tgt_tmp == tgt_padding_num.unsqueeze(1)).type_as(src_tokens)

                back_data["src_lengths"] = (prev_output_tokens != 1).sum(dim=1)
                back_data["src_tokens"] = (
                    torch.cat(
                        (
                            prev_output_tokens[:, 1:],
                            torch.ones((bsz, 1), device="cuda", dtype=torch.long),
                        ),
                        dim=1,
                    )
                    + torch.ones(
                        prev_output_tokens.size(), device="cuda", dtype=torch.long
                    )
                    * src_tmp
                )
                _src_tokens = src_tokens[:, :-1].contiguous()
                _src_tokens[_src_tokens == 2] = 1
                back_data["prev_output_tokens"] = torch.cat(
                    (
                        torch.full((bsz, 1), 2, device="cuda", dtype=torch.long),
                        _src_tokens,
                    ),
                    dim=1,
                )
                return back_data

            back_data = produce_groundtruth_back_data(
                sample["net_input"]["src_tokens"],
                sample["net_input"]["src_lengths"],
                sample["net_input"]["prev_output_tokens"],
            )
            back_sample["net_input"] = back_data
            return back_sample

        # sample=make_back_sample(sample)
        gen_timer.start()
        hypos, g, src_lens = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)
        rws.extend([d201(g[i], src_lens[i]) for i in range(len(g))])

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                    sample_id
                )
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                # pdb.set_trace()
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )
                    if cfg.generation.print_alignment == "soft":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [",".join(src_probs) for src_probs in alignment]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if (
                        align_dict is not None
                        or cfg.common_eval.post_process is not None
                    ):
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
            ),
            file=output_file,
        )

    cw, ap, al, dal = compute_delay(rws, is_weight_ave=True)
    print("CW score: ", cw)
    print("AP score: ", ap)
    print("AL score: ", al)
    print("DAL score: ", dal)
    return scorer


def d201(d, src):
    # print("+++",d)
    s = "0 " * int(d[0] + 1) + "1 "
    for i in range(1, len(d)):
        s = s + "0 " * int((min(d[i], src - 1) - min(d[i - 1], src - 1))) + "1 "
    if src > d[-1] + 1:
        s = s + "0 " * (src - d[-1] - 1)
    return s


def generate_rw(src_len, tgt_len, k):
    rws = []
    gs = []
    for i in range(0, len(src_len)):
        g = ""
        for j in range(0, tgt_len[i] - 1):
            g += str(min(k + j, src_len[i] - 1)) + " "
        if src_len[i] <= k:
            rw = "0 " * src_len[i] + "1 " * tgt_len[i]
        else:
            if tgt_len[i] + k > src_len[i]:
                rw = (
                    "0 " * k
                    + "1 0 " * (src_len[i] - k)
                    + "1 " * (tgt_len[i] + k - src_len[i])
                )
            else:
                rw = (
                    "0 " * k
                    + "1 0 " * (tgt_len[i])
                    + "0 " * (src_len[i] - tgt_len[i] - k)
                )
        rws.append(rw)
        gs.append(g)
    # print(rws)
    return rws, gs


def compute_delay(rw, is_weight_ave=False):
    CWs, ALs, APs, DALs, Lsrc = [], [], [], [], []
    for line in rw:
        line = line.strip()
        al_ans = RW2AL(line, add_eos=False)
        dal_ans = RW2DAL(line, add_eos=False)
        ap_ans = RW2AP(line, add_eos=False)
        cw_ans = RW2CW(line, add_eos=False)
        if al_ans is not None:
            ALs.append(al_ans)
            DALs.append(dal_ans)
            APs.append(ap_ans)
            CWs.append(cw_ans)
            Lsrc.append(line.count("0"))

    CW = np.average(CWs) if is_weight_ave else np.average(CWs, weights=Lsrc)
    AL = np.average(ALs) if is_weight_ave else np.average(ALs, weights=Lsrc)
    DAL = np.average(DALs) if is_weight_ave else np.average(DALs, weights=Lsrc)
    AP = np.average(APs) if is_weight_ave else np.average(APs, weights=Lsrc)
    return CW, AP, AL, DAL


def RW2CW(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0
    c = s.count("01")

    if c == 0:
        return 0
    else:
        return x / c


# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AP(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    curr = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
    return sum(curr) / x / y


# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AL(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    rate = y / x
    curr = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
        if i == "1" and count == x:
            break
    y1 = len(curr)
    diag = [(t - 1) / rate for t in range(1, y1 + 1)]
    return sum(l1 - l2 for l1, l2 in zip(curr, diag)) / y1


def RW2DAL(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    rate = y / x
    curr = []
    curr1 = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
    curr1.append(curr[0])
    for i in range(1, y):
        curr1.append(max(curr[i], curr1[i - 1] + 1 / rate))

    diag = [(t - 1) / rate for t in range(1, y + 1)]
    return sum(l1 - l2 for l1, l2 in zip(curr1, diag)) / y


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
