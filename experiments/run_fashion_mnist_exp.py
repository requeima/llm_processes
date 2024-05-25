from run_llm_process import run_llm_process
from hf_api import get_model_and_tokenizer, llm_map
from parse_args import init_option_parser
from jsonargparse import ArgumentParser


images = [105, 294, 411, 436, 482, 485]
sizes = [20, 50]
gen_length = 6


def main():
    # parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--llm_path', type=str, help='Path to LLM.')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument("--llm_type", choices=llm_map.keys(), default="llama-2-7B",
                        help="Hugging face model to use.")
    parser.add_argument('--output_dir', type=str, default='./output/fashion_mnist/',
                        help='Path to directory where output results are written.')
    parser.add_argument('--plot_dir', type=str, default='./plots/fashion_mnist/',
                        help='Path to directory where output plots are written.')
    args = parser.parse_args()
    print(args, flush=True)

    option_parser = init_option_parser()

    model, tokenizer = get_model_and_tokenizer(args.llm_path, args.llm_type)
    for image in images:
        for size in sizes:
            config_args = option_parser.parse_args(args=[
                    "--experiment_name", "fashion_mnist_{}_image_{}_size_{}".format(args.llm_type, image, size),
                    "--data_path", "./data/images/fashion_mnist_{}_{}.pkl".format(image, size),
                    "--output_dir", args.output_dir,
                    "--plot_dir", args.plot_dir,
                    "--batch_size", str(args.batch_size),
                    "--max_generated_length", str(gen_length),
                    "--num_decimal_places_x", str(0),
                    "--num_decimal_places_y", str(0),
                    "--num_samples", str(args.num_samples),
            ])
            print(config_args, flush=True)
            run_llm_process(args=config_args, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()

