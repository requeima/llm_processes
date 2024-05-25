import torch
import torch.nn.functional as F
import numpy as np
import pickle
import math
import os
from functools import partial
from helpers import construct_prompts, floats_to_str, my_removesuffix
from tqdm import tqdm


def _get_ranges(decode_fn, enc_full_text, text_blocks):
    '''
    Returns index ranges corresponding to texts in text_blocks in token space.

    Args:
        decode_fn: tokenizer's decoding function.
        enc_full_text: list of tokens-- the result of calling
            tokenizer.encode(full_text).
        text_blocks: list of strings [text_1, ..., text_n]
            such that ''.join(text_blocks) == full_text.
            We assume that the texts are non_ys and ys in alternating fashion:
            text_blocks = [non_y_1, y_1, ..., non_y_m, y_m].
    Returns:
        y_ranges: list of tuples [y_range_1, ..., y_range_m] such that
            enc_full_text[y_range_i[0]: y_range_i[1]] == f'{y_i}{break_str}',
            essentially isolating just the y portions of the full_text.
        non_y_ranges: list of tuples [non_y_range_1, ..., non_y_range_m]
            such that 
            enc_full_text[non_y_range_i[0]: non_y_range_i[1]] == non_y_i.
    '''
    ranges = []
    curr_block = 0
    tok_left = 0  # pointer in the token space
    left = 0      # pointer in the text space
    for tok_right in range(1, len(enc_full_text) + 1):
        dec_text = decode_fn(enc_full_text[: tok_right])
        if dec_text[left : ] == text_blocks[curr_block]:
            ranges.append((tok_left, tok_right))
            tok_left = tok_right
            left = len(dec_text)
            curr_block += 1

    assert len(ranges) == len(text_blocks)
    non_y_ranges, y_ranges = [], []
    for i, range_i in enumerate(ranges):
        y_ranges.append(range_i) if i % 2 else non_y_ranges.append(range_i)
    return y_ranges, non_y_ranges


def _get_mask(model, allowed_tokens):
    out_size = model.get_output_embeddings().out_features
    mask = torch.ones(out_size, dtype=torch.bool, device='cuda')
    mask[allowed_tokens] = False
    return mask


@torch.inference_mode()
def _get_y_logprobs(args, tokenizer, model, input_tokens, mask, y_ranges):
    '''
    Gets the logprobs of the y portions in input_texts,
    where the positions of y are given by y_ranges.

    Args:
        input_tokens: list (of length N) of encodings.
        y_ranges: list of y_ranges [y_range_1, ..., y_range_N].
    Returns:
        list of logprobs for each y given by y_ranges
            [y_logprobs_1, ..., y_logprobs_N], 
            where each y_logprobs_i is an np.array.
    '''
    bs = len(input_tokens)
    max_prompt_len = max(len(t) for t in input_tokens)

    input_ids = torch.full((bs, max_prompt_len),
                        tokenizer.pad_token_id,
                        dtype=torch.long, device='cuda')
    attn_mask = torch.zeros(
        (bs, max_prompt_len), dtype=torch.long, device='cuda')
    for k, t in enumerate(input_tokens):
        input_ids[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cuda')
        attn_mask[k, : len(t)] = torch.ones(len(t), dtype=torch.long, device='cuda')

    # Shift so that tokens < n predict n
    outputs = model(input_ids=input_ids[:, :], attention_mask=attn_mask[:, :])
    shift_logits = outputs['logits'][..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    if mask is not None:
        shift_logits[:, :,  mask] = -100
    logprobs = -F.cross_entropy(input=shift_logits.transpose(1, 2), target=shift_labels, reduction='none')

    if args.print_logprobs:
        log_probs = []
        log_probs.append((tokenizer.decode(input_ids[0].tolist()[0]), None))
        for idx, (label_id, logit) in enumerate(zip(shift_labels[0].tolist(), shift_logits[0])):
                logprob = torch.nn.functional.log_softmax(logit, dim=0).tolist()[label_id]
                log_probs.append((tokenizer.decode(label_id), float(logprob)))
        print(log_probs)

    y_logprobs = [logprobs[i][y_ranges[i][0] - 1 : y_ranges[i][1] - 1].cpu()
                for i in range(bs)]
    return y_logprobs


def compute_nll(args, tokenizer, model, results):
    if args.specify_xy and (args.xs is None) and (args.ys == None):
        # generate the xs and ys
        assert args.xs_start is not None
        assert args.xs_end is not None
        assert args.num_xs is not None
        assert args.ys_start is not None
        assert args.ys_end is not None
        assert args.num_ys is not None
        args.xs = (np.linspace(start=args.xs_start, stop=args.xs_end, num=args.num_xs, endpoint=True)).tolist()
        args.ys = (np.linspace(start=args.ys_start, stop=args.ys_end, num=args.num_ys, endpoint=True)).tolist()

    results['dist'] = []
    results['y_logprobs'] = []
    if args.specify_xy:
        results['xs'] = args.xs
        results['ys'] = args.ys

    get_prompts = partial(construct_prompts, 
                        prefix=args.prefix,
                        x_prefix=args.x_prefix,
                        y_prefix=args.y_prefix,
                        break_str=args.break_str,
                        dim_x=results['dim_x'],
                        dim_y=results['dim_y'],
                        num_decimal_x=args.num_decimal_places_x,
                        num_decimal_y=args.num_decimal_places_y,
                        order=args.prompt_ordering,
                        x_ordering=results['data']['x_ordering'] if 'x_ordering' in results['data'] else None,)

    space = ' ' if args.y_prefix.endswith(' ') else ''
    decode_fn = partial(tokenizer.decode, skip_special_tokens=True)
    # There are two cases:
    # 1) Given x_test_i, evaluate the log prob of y_true_i.
    #   a) Autoregressively: evalute (x_test_i, y_true_i) with (x_test_j, y_true_j)
    #      where j < i prepended to the prompt.
    #   b) Marginally: evaluate each (x_test_i, y_true_i) independently.
    # 2) Given x_test_i, evaluate the log prob of each y in args.ys.
    # For both cases, we find the index ranges for the y tokens
    # in order to extract the logprob of just the y portion of the prompts.
    y_tokens = [] # Keep in track of all tokens that consist the y strings.
    full_texts, enc_full_texts, y_ranges = [], [], []
    max_len = 0
    if not args.specify_xy:
        if args.autoregressive:
            x_context, y_context = results['data']['x_train'], results['data']['y_train']
            for x_test, y_test_true in tqdm((zip(results['data']['x_test'], results['data']['y_test'])), desc='Processing prompts'):
                # The rstrip and prepending y with a space is necessary because
                # tokenization lumps a space in front of the negative sign,
                # so the non-y portion ends up without a space when y is negative.
                # So to be consistent between positive and negative numbers,
                # non_y doesn't have a space and y includes a space.
                non_y = get_prompts(x_train=x_context, y_train=y_context,
                                    x_test=np.array([x_test]))[0]
                str_y_test_true = floats_to_str(y_test_true, args.num_decimal_places_y)
                y = f'{str_y_test_true}{args.break_str}'
                # if args.llm_type == 'gemma':
                #     y = space + y
                full_text = non_y + y
                enc_full_text = tokenizer.encode(full_text)

                enc_non_y = tokenizer.encode(my_removesuffix(non_y, ' '))
                y_range = [len(enc_non_y), len(enc_full_text)]

                if results['dim_x'] > 1:
                    x_context = np.append(x_context, np.expand_dims(x_test, axis=0), axis=0)
                else:
                    x_context = np.append(x_context, x_test)
                if results['dim_y']> 1:
                    y_context = np.append(y_context, np.expand_dims(y_test_true, axis=0), axis=0)
                else:
                    y_context = np.append(y_context, y_test_true)

                full_texts.append(full_text)
                enc_full_texts.append(enc_full_text)
                y_ranges.append(y_range)
                y_tokens.append(np.unique(enc_full_text[y_range[0]: y_range[1]]))
                max_len = max(max_len, len(enc_full_text))
        else:
            for x_test, y_test_true in tqdm((zip(results['data']['x_test'], results['data']['y_test'])), desc='Processing prompts'):
                non_y = get_prompts(x_train=results['data']['x_train'], y_train=results['data']['y_train'],
                                    x_test=np.array([x_test]))[0]
                str_y_test_true = floats_to_str(y_test_true, args.num_decimal_places_y)
                y = f'{str_y_test_true}{args.break_str}'
                full_text = non_y + y
                enc_full_text = tokenizer.encode(full_text)

                enc_non_y = tokenizer.encode(my_removesuffix(non_y, ' '))
                y_range = [len(enc_non_y), len(enc_full_text)]

                full_texts.append(full_text)
                enc_full_texts.append(enc_full_text)
                y_ranges.append(y_range)
                y_tokens.append(np.unique(enc_full_text[y_range[0]: y_range[1]]))
                max_len = max(max_len, len(enc_full_text))
    else:
        for i, x_test in enumerate(args.xs):
            full_texts.append([])
            enc_full_texts.append([])
            y_ranges.append([])
            non_y_range = None
            for j, y_test in enumerate(args.ys):
                non_y = get_prompts(x_train=results['data']['x_train'], y_train=results['data']['y_train'],
                                    x_test=np.array([x_test]))[0].rstrip(' ')
                str_y_test = floats_to_str(y_test, args.num_decimal_places_y)
                y = f'{space}{str_y_test}{args.break_str}'
                full_text = non_y + y
                enc_full_text = tokenizer.encode(full_text)

                if non_y_range is None:
                    _, non_y_range = _get_ranges(decode_fn, enc_full_text, [non_y, y])
                    assert len(non_y_range) == 1
                    non_y_range = non_y_range[0]
                y_range = (non_y_range[1], len(enc_full_text))

                full_texts[-1].append(full_text)
                enc_full_texts[-1].append(enc_full_text)
                y_ranges[-1].append(y_range)
                y_tokens.append(np.unique(enc_full_text[y_range[0]: y_range[1]]))
                max_len = max(max_len, len(enc_full_text))
    results['full_texts'] = full_texts
    results['enc_full_texts'] = enc_full_texts
    results['y_ranges'] = y_ranges
    unique_y_tokens = set(np.unique(np.concatenate(y_tokens)).tolist())

    # Generate mask optionally.
    mask = None
    if args.mask_unused_tokens:
        # When y is a number, it can also include '-', and '.'.
        allowed = [str(i) for i in range(10)] + ['-', '.']
        allowed_tokens = set([tokenizer.convert_tokens_to_ids(token)
                            for token in allowed])
        diff = unique_y_tokens - allowed_tokens
        for t in diff:
            allowed_tokens.add(t)
        mask = _get_mask(model, list(allowed_tokens))

    if not args.specify_xy:
        num_batches = math.ceil(len(enc_full_texts) / args.batch_size)
        y_logprobs = []
        for i in tqdm(range(num_batches), desc="Computing log probs"):
            y_logprobs.extend(_get_y_logprobs(args, tokenizer, model,
                enc_full_texts[i * args.batch_size : (i + 1) * args.batch_size],
                mask,
                y_ranges[i * args.batch_size : (i + 1) * args.batch_size]))

        nll = 0
        for y_lp in y_logprobs:
            # log p(x) = log p(token) - log bin_width
            nll += -(y_lp.sum().item() + results['dim_y'] * args.num_decimal_places_y * np.log(10))
            results['y_logprobs'].append(y_lp.to(float).numpy())
        results['nll'] = nll
        results['avg_nll'] = nll / len(y_logprobs)
        print("avg_nll = {}".format(results['avg_nll']))
    else:
        num_batches = math.ceil(len(full_texts[0]) / args.batch_size) # this is the number of batches per test location
        for itr in tqdm(range(len(results['dist']), len(full_texts)), desc="Computing log probs"): # this loops over number of test locations
            y_logprobs_given_x = []
            for i in range(num_batches):
                y_logprobs = _get_y_logprobs(args, tokenizer, model,
                    enc_full_texts[itr][i * args.batch_size : (i + 1) * args.batch_size],
                    mask,
                    y_ranges[itr][i * args.batch_size : (i + 1) * args.batch_size])
                y_logprobs_given_x.extend(y_logprobs)

            y_probs_given_x = [np.exp(y_lp.sum().item()) for y_lp in y_logprobs_given_x]
            results['dist'].append(y_probs_given_x / np.sum(y_probs_given_x))
            results['y_logprobs'].append([y_lp.to(float).numpy() for y_lp in y_logprobs_given_x])

    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(results, f)

    return results