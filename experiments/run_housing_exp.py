from jsonargparse import ArgumentParser
from llm_processes.run_llm_process import run_llm_process
from llm_processes.hf_api import get_model_and_tokenizer, llm_map
from llm_processes.parse_args import init_option_parser


prompt = "The following is a dataset comprising housing prices and various variables around housing and demographics for the top 50 American cities by population.\n"

def main():
    # parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--llm_path', type=str, help='Path to LLM.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--llm_type", choices=llm_map.keys(), default="llama-2-7B",
                        help="Hugging face model to use.")
    parser.add_argument("--autoregressive", type=bool, default=False,
                        help="If true, append the previous prediction to the current prompt.")
    parser.add_argument("--max_generated_length",  type=int, default=16)
    parser.add_argument("--sort_x_test",  type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./output/housing',
                        help='Path to directory where output results are written.')
    parser.add_argument('--plot_dir', type=str, default='./plots/housing',
                        help='Path to directory where output plots are written.')    

    args = parser.parse_args()
    print(args, flush=True)

    option_parser = init_option_parser()

    model, tokenizer = get_model_and_tokenizer(args.llm_path, args.llm_type)

    # Experiment 1    
    # Loop for all the samples and features
    for j in range(10):
        for i in range(4):
            print('features:', i, 'sample:', j)
            config_args = option_parser.parse_args(args=[
                    "--experiment_name", "{}_housing_prices_sample_{}_features_{}_auto_{}_exp_1".format(args.llm_type, j, i, args.autoregressive),
                    "--data_path", "./data/housing/American_Housing_Data_20231209_sample_{}_features_{}.pkl".format(j,i),
                    "--output_dir", args.output_dir,
                    "--plot_dir", args.plot_dir,
                    "--batch_size", str(args.batch_size),
                    "--prefix", prompt,
                    "--autoregressive", str(args.autoregressive),
                    "--max_generated_length", str(args.max_generated_length),
                    "--sort_x_test", str(args.sort_x_test),
                    "--num_samples", str(args.num_samples),
                    '--y_prefix', ' Price: ',
                    "--prompt_ordering", "random",
                    ])
            
            run_llm_process(args=config_args, model=model, tokenizer=tokenizer)
            
    # Experiment 2
    # Loop for all the samples and features
    for j in range(10):
        for i in range(4):
            print('features:', i, 'sample:', j)
            config_args = option_parser.parse_args(args=[
                    "--experiment_name", "{}_housing_prices_sample_{}_features_{}_auto_{}_exp_2".format(args.llm_name, j, i, args.autoregressive),
                    "--data_path", "./data/housing/American_Housing_Data_20231209_sample_{}_features_{}_exp_2.pkl".format(j,i),
                    "--output_dir", args.output_dir,
                    "--plot_dir", args.plot_dir,
                    "--batch_size", str(args.batch_size),
                    "--prefix", prompt,
                    "--autoregressive", str(args.autoregressive),
                    "--max_generated_length", str(args.max_generated_length),
                    "--sort_x_test", str(args.sort_x_test),
                    "--num_samples", str(args.num_samples),
                    '--y_prefix', ' Price: ',
                    "--prompt_ordering", "random",
                    ])




if __name__ == '__main__':
    main()
