import torch

# Core code for ACL 2022 paper "Modeling Dual Read/Write Paths for Simultaneous Machine Translation"


def generate_dual_path(alpha):
    # Generate Dual Path (GDP fuction)
    bsz, tgt_len, src_len = alpha.size()

    # Segment
    max_idx = alpha.max(dim=-1, keepdim=True)[1]
    inn = torch.cat(
        (
            torch.full((max_idx.size(0), 1, max_idx.size(2)), -1, device=alpha.device),
            max_idx[:, :-1, :],
        ),
        dim=1,
    )
    min_idx = (max_idx > inn).int()
    inn += 1
    min_idx = min_idx * inn
    min_idx = torch.cummax(min_idx, dim=1)[0]
    tmp = (
        torch.arange(0, src_len, device=alpha.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bsz, tgt_len, 1)
    )
    mask1 = (tmp <= max_idx).int()
    mask2 = (tmp >= min_idx).int()
    mask = mask1 * mask2

    # Transpose
    mask = mask.transpose(1, 2)

    # Merge
    tmp = (
        torch.arange(0, tgt_len, device=alpha.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bsz, src_len, 1)
    )
    mask = (tmp <= mask.max(dim=-1, keepdim=True)[1]).int() + mask
    mask = mask.bool().int()
    # Ensure the read/write path is monotonic and valid
    src_lens = mask.sum(dim=2, keepdim=True)
    src_lens = torch.cummax(src_lens, dim=1)[0]
    dual_path = (tmp == src_lens - 1).int()
    dual_path = dual_path / dual_path.sum(dim=-1, keepdim=True)

    return dual_path


def process_back_data(src_tokens, src_lengths, prev_output_tokens):
    # prepare the data for the target-to-source SiMT
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
        src_tmp == (prev_output_tokens.size(1) - src_padding_num - 1).unsqueeze(1)
    ).type_as(src_tokens)

    tgt_tmp = (
        torch.arange(0, src_tokens.size(1), device="cuda").unsqueeze(0).repeat(bsz, 1)
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
        + torch.ones(prev_output_tokens.size(), device="cuda", dtype=torch.long)
        * src_tmp
    )

    _src_tokens = src_tokens[:, :-1].contiguous()
    _src_tokens[_src_tokens == 2] = 1
    back_data["prev_output_tokens"] = torch.cat(
        (torch.full((bsz, 1), 2, device="cuda", dtype=torch.long), _src_tokens),
        dim=1,
    )

    return back_data
