import os
from run_llm_process import run_llm_process
from hf_api import get_model_and_tokenizer, llm_map
from parse_args import init_option_parser
from jsonargparse import ArgumentParser

gen_length = 7

sizes = [10, 20, 30, 40, 50]


def main():
    # parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument("--llm_type", choices=llm_map.keys(), default="llama-2-7B",
                        help="Hugging face model to use.")
    parser.add_argument('--llm_path', type=str, help='Path to LLM.')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument("--hf_model_kind", choices=["llama-2", "llama-3", "mixtral", "phi-3"], default="llama-2",
                        help="Hugging face model to use.")
    args = parser.parse_args()
    print(args, flush=True)

    option_parser = init_option_parser()

    model, tokenizer = get_model_and_tokenizer(args.llm_path, args.llm_type)
    for size in sizes:
        config_args = option_parser.parse_args(args=[
                "--experiment_name", "compare_{}_llm_proc_auto_{}".format(args.llm_type, size),
                "--data_path", os.path.join("./data/weather", "weather_llm_proc_{}".format(size) + '.pkl'),
                "--output_dir", "./output/compare",
                "--plot_dir", "./plots/compare",
                "--batch_size", str(args.batch_size),
                "--max_generated_length", str(gen_length),
                "--autoregressive", "True"
        ])
        run_llm_process(args=config_args, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()