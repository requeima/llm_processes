import numpy as np
import os


from .plot import plot_samples, plot_images, plot_heatmap
from .hf_api import get_model_and_tokenizer
from .parse_args import parse_command_line
from .compute_nll import compute_nll
from .sample import sample
from .prepare_data import prepare_data


def run_llm_process(args, model, tokenizer):
    np.random.seed(args.seed)

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    results = prepare_data(args)

    if 'sample' in args.mode:
        results = sample(args, tokenizer, model, results)
    
    if 'logpy' in args.mode:
        results = compute_nll(args, tokenizer, model, results)

    if 'sample' in args.mode:
        # plot results
        if ('mnist' in args.experiment_name):
            plot_images(results, args.experiment_name, args.plot_dir)
        else:
            plot_samples(results, args.experiment_name, args.plot_trajectories, args.plot_dir)
    if args.specify_xy:
        plot_heatmap(results, args.experiment_name, args.plot_dir, args.xs, args.ys)


def main():
    # parse the command line arguments
    args = parse_command_line()

    # get the llm and asociated tokenizer
    model, tokenizer = get_model_and_tokenizer(args.llm_path, args.llm_type)

    run_llm_process(args=args, model=model, tokenizer=tokenizer)
