import torch
import os
import pickle
import numpy as np
from tqdm import tqdm

from .helpers import construct_prompts, get_num_from_gen, process_generated_results
from .hf_api import hf_generate_batch, hf_generate

def sample(args, tokenizer, model, results):   
    with torch.no_grad():
        # generate
        results['gen'] = [[] for _ in range(len(results['data']['x_test']))]
        if args.autoregressive:
            x_train_current = [results['data']['x_train']] * args.num_samples
            y_train_current = [results['data']['y_train']] * args.num_samples

            prompts = [[] for _ in range(len(results['data']['x_test']))]
            for idx, x in tqdm(enumerate(results['data']['x_test']), desc='Sampling'):
                samples = [[] for _ in range(args.num_samples)]
                per_sample_prompts = [[] for _ in range(args.num_samples)]
                sample_indices_to_process = np.arange(0, args.num_samples).tolist()
                while len(sample_indices_to_process) > 0:
                    batch_size = min(args.batch_size, len(sample_indices_to_process))
                    batch_indices = sample_indices_to_process[0 : batch_size]
                    batch_prompts = []
                    for sample_index in batch_indices:
                        prompt = construct_prompts(
                            x_train=x_train_current[sample_index],
                            y_train=y_train_current[sample_index],
                            x_test=np.array([x]),
                            prefix=args.prefix,
                            x_prefix=args.x_prefix,
                            y_prefix=args.y_prefix,
                            break_str=args.break_str,
                            dim_x=results['dim_x'],
                            dim_y=results['dim_y'],
                            num_decimal_x=args.num_decimal_places_x,
                            num_decimal_y=args.num_decimal_places_y,
                            order=args.prompt_ordering,
                            add_spaces=False,
                            x_ordering=results['data']['x_ordering'] if 'x_ordering' in results['data'] else None,
                        )
                        batch_prompts.append(prompt[0])

                    res = hf_generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch_prompts,
                        temp=args.temperature, 
                        top_p=args.top_p,
                        max_new_tokens=args.max_generated_length
                    )
                    assert(len(res) == len(batch_indices))
                    for i, sample_index in enumerate(batch_indices):
                        generated = get_num_from_gen(
                            gen=res[i],
                            break_str=args.break_str,
                            dim_y=results['dim_y'],
                            max_generated_length=args.max_generated_length,
                            num_decimal_places_y=args.num_decimal_places_y
                        )
                        if generated is not None:
                            if results['dim_x'] > 1:
                                x_train_current[sample_index] = np.append(x_train_current[sample_index], np.expand_dims(np.array(x), axis=0), axis=0)
                            else:
                                x_train_current[sample_index] = np.append(x_train_current[sample_index], np.array(x))
                            if results['dim_y'] > 1:
                                y_train_current[sample_index] = np.append(y_train_current[sample_index], np.expand_dims(np.array(generated), axis=0), axis=0)
                            else:
                                y_train_current[sample_index] = np.append(y_train_current[sample_index], np.array(generated))
                            sample_indices_to_process.remove(sample_index)
                            per_sample_prompts[sample_index] = batch_prompts[i]
                            samples[sample_index] = res[i]
                results['gen'][idx] += samples
                prompts[idx] += per_sample_prompts
            # Print out the first sample.
            if args.print_prompts:
                for prompt, gen in zip(prompts, results['gen']):
                    print(prompt[0], flush=True)
                    print(f"> {gen[0]}", flush=True)
                    print("\n==================================\n", flush=True)
        else:
            # generate the prompts from the data
            prompts = construct_prompts(
                x_train=results['data']['x_train'],
                y_train=results['data']['y_train'],
                x_test=results['data']['x_test'],
                prefix=args.prefix,
                x_prefix=args.x_prefix,
                y_prefix=args.y_prefix,
                break_str=args.break_str,
                dim_x=results['dim_x'],
                dim_y=results['dim_y'],
                num_decimal_x=args.num_decimal_places_x,
                num_decimal_y=args.num_decimal_places_y,
                order=args.prompt_ordering,
                add_spaces=False,
                x_ordering=results['data']['x_ordering'] if 'x_ordering' in results['data'] else None,
            )

            num_prompts = len(prompts)
            for idx in tqdm(range(num_prompts), desc='Sampling'):
                prompt = prompts[idx]
                samples = []
                num_samples = args.num_samples
                while num_samples > 0:
                    bs = min(args.batch_size, num_samples)
                    res = hf_generate(
                        model=model,
                        tokenizer=tokenizer,
                        input_str=prompt,
                        batch_size=bs,
                        temp=args.temperature, 
                        top_p=args.top_p,
                        max_new_tokens=args.max_generated_length
                    )
                    for j in range(len(res)):
                        if get_num_from_gen(
                            gen=res[j],
                            break_str=args.break_str,
                            dim_y=results['dim_y'],
                            max_generated_length=args.max_generated_length,
                            num_decimal_places_y=args.num_decimal_places_y
                            ) is not None:
                            samples.append(res[j])
                            num_samples -= 1
                results['gen'][idx] += samples

            # Print out the first sample.
            if args.print_prompts:
                for prompt, gen in zip(prompts, results['gen']):
                    print(prompt, flush=True)
                    print(f"> {gen[0]}", flush=True)
                    print("\n==================================\n", flush=True)

    results['prompts'] = prompts

    results = process_generated_results(
        gen_results=results,
        break_str=args.break_str,
        dim_y=results['dim_y'],
        max_generated_length=args.max_generated_length,
        num_decimal_places_y=args.num_decimal_places_y
    )

    # save off the results
    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(results, f)

    return results