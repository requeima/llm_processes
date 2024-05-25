import optuna
import pickle
import os
import numpy as np
from experiments.classics import function_map
from run_llm_process import run_llm_process
from hf_api import get_model_and_tokenizer, llm_map
from parse_args import init_option_parser
from jsonargparse import ArgumentParser


def main():
    # parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--experiment_name_prefix', type=str, help='Name of the experiment.')
    parser.add_argument('--data_dir', type=str, default='./output/black_box/data',
                        help='Path to pkl file with x, y data.')
    parser.add_argument('--llm_path', type=str, help='Path to LLM.')
    parser.add_argument("--prompt_ordering", choices=["sequential", "random", "distance"], default="distance",
                        help="How the observed points in the prompt should be ordered.")
    parser.add_argument('--output_dir', type=str, default='./output/black_box',
                        help='Path to directory where output results are written.')
    parser.add_argument('--plot_dir', type=str, default='./plots/black_box',
                        help='Path to directory where output plots are written.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_decimal_places_x', type=int, default=2)
    parser.add_argument('--num_decimal_places_y', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_generated_length', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--llm_type", choices=llm_map.keys(), default="llama-2-7B",
                        help="Hugging face model to use.")
    parser.add_argument("--autoregressive", type=bool, default=False,
                        help="If true, append the previous prediction to the current prompt.")
    parser.add_argument('--prefix', type=str, default='', help='Prompt prefix.')
    parser.add_argument('--x_prefix', type=str, default='', help='Prompt x prefix.')
    parser.add_argument('--y_prefix', type=str, default=', ', help='Prompt y prefix.')
    parser.add_argument('--z_prefix', type=str, default=', ', help='Prompt z prefix.')
    parser.add_argument('--break_str', type=str, default='\n', help='Break string between observed points.')
    parser.add_argument("--function", choices=function_map.keys(), default="Sinusoidal",
                        help="Function to optimize.")
    parser.add_argument('--num_cold_start_points', type=int, default=5)
    parser.add_argument('--num_test_points', type=int, default=500)
    parser.add_argument('--num_true_points', type=int, default=500)
    parser.add_argument('--num_trials', type=int, default=100)
    args = parser.parse_args()
    print(args, flush=True)

    np.random.seed(args.seed)

    function = function_map[args.function]
    bounds = function.bounds
    x_min, x_max = bounds.T
    ndim = function.ndim
    function_max = function.get_f(function.xopt)

    # optuna
    def objective(trial):
        x = []
        for i in range(ndim):
            x.append(trial.suggest_float('x' + str(i), x_min[i], x_max[i]))
        return function.get_f(x)

    study = optuna.create_study(study_name=args.function, direction="maximize")
    study.optimize(objective, n_trials=args.num_trials)
    print(study.best_params)

    # llm_proc
    # create the necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # get the llm and asociated tokenizer
    model, tokenizer = get_model_and_tokenizer(args.llm_path, args.llm_type)
    option_parser = init_option_parser()

    x_trial = []
    y_trial = []

    # train
    # pick some points at random for cold start
    if ndim == 1:  # equally spaced in 1D
        x_train = np.linspace(start=x_min, stop=x_max, num=args.num_cold_start_points)
    else: # random in higher D
        x_train = []
        for i in range(ndim):
            x_train.append(np.random.uniform(low=x_min[i], high=x_max[i], size=args.num_cold_start_points))
        x_train = (np.array(x_train)).T
    y_train = function.get_f(x_train)

    # test
    if ndim == 1:
        x_test = np.linspace(start=x_min, stop=x_max, num=args.num_test_points)
    else:
        x_test = []
        for i in range(ndim):
            x_test.append(np.random.uniform(low=x_min[i], high=x_max[i], size=args.num_test_points))
        x_test = (np.array(x_test)).T
    y_test = function.get_f(x_test)
    
    # true
    if ndim == 1:
        x_true = np.linspace(start=x_min, stop=x_max, num=args.num_true_points)
    else:
        x_true = []
        for i in range(ndim):
            x_true.append(np.random.uniform(low=x_min[i], high=x_max[i], size=args.num_true_points))
        x_true = (np.array(x_true)).T        
    y_true = function.get_f(x_true)

    data = {
        'x_train': x_train.squeeze(),  # squeeze for the 1D case
        'y_train': y_train,
        'x_test': x_test.squeeze(),  # squeeze for the 1D case
        'y_test': y_test,
        'x_true': x_true.squeeze(),  # squeeze for the 1D case
        'y_true': y_true        
    }

    # copy the cold start training points into the trial list
    for trial, (x, y) in enumerate(zip(x_train, y_train)):
        x_trial.append(x)
        y_trial.append(y)
        print("Cold start point {}".format(trial), flush=True)
        print("Trial {} finished with value: {} and parameters {}. Best is trial {} with value {}. Target is {}.".format(trial, y, x, np.argmax(np.array(y_trial)), np.max(np.array(y_trial)), function_max), flush=True)

    for trial in range(args.num_cold_start_points, args.num_trials):
        # get the next point
        if (trial % 4) == 3:  # do some exploring every 4th trial
            new_training_x = []
            for i in range(ndim):
                new_training_x.append(np.random.uniform(low=x_min[i], high=x_max[i], size=1))
            new_training_x = (np.array(new_training_x)).T
            new_training_x = new_training_x.squeeze()
            x_trial.append(new_training_x)
            y_trial.append(function.get_f(new_training_x).squeeze())
            print("Exploring", flush=True)
        else:  # Sample the LLM with the current training points and Thompson sample a new point.
            experiment_name = "{}_trial_{:02d}".format(args.experiment_name_prefix, trial)
            data_file_path = os.path.join(args.data_dir, experiment_name + '.pkl')
            with open(data_file_path,  "wb") as f:
                pickle.dump(data, f)

            print("Thompson Sampling", flush=True)
            config_args = option_parser.parse_args(args=[
                    "--mode", "sample_only",
                    "--experiment_name", experiment_name,
                    "--data_path", data_file_path,
                    "--llm_path", args.llm_path,
                    "--output_dir", args.output_dir,
                    "--plot_dir", args.plot_dir,
                    "--num_samples", str(args.num_samples),
                    "--batch_size", str(args.batch_size),
                    "--num_decimal_places_x", str(args.num_decimal_places_x),
                    "--num_decimal_places_y", str(args.num_decimal_places_y),
                    "--top_p", str(args.top_p),
                    "--temperature", str(args.temperature),
                    "--max_generated_length", str(args.max_generated_length),
                    "--prompt_ordering", args.prompt_ordering,
                    "--autoregressive", str(args.autoregressive),
            ])
            run_llm_process(args=config_args, model=model, tokenizer=tokenizer)

            # get the results of the sampling
            with open(os.path.join(args.output_dir, experiment_name + '.pkl'),  "rb") as f:
                results = pickle.load(f)

            x_trial.append(results['y_test_max_x'])
            y_trial.append(function.get_f(results['y_test_max_x']).squeeze())

            new_training_x = results['y_test_max_x']

        new_training_y = function.get_f(new_training_x)
        print("New training point is ({}, {})".format(new_training_x, new_training_y), flush=True)

        print("Trial {} finished with value: {} and parameters {}. Best is trial {} with value {}. Target is {}.".format(trial, new_training_y, new_training_x, np.argmax(np.array(y_trial)), np.max(np.array(y_trial)), function_max), flush=True)

        if ndim == 1:
            x_train = np.append(x_train, new_training_x)
        else:
            new_training_x = np.expand_dims(new_training_x, axis=0)
            x_train = np.append(x_train, new_training_x, axis=0)
        y_train = np.append(y_train, new_training_y)

        data['x_train'] = x_train
        data['y_train'] = y_train

        # generate a new set of test points for the next trial
        if ndim == 1:
            x_test = np.linspace(start=x_min, stop=x_max, num=args.num_test_points)
        else:
            x_test = []
            for i in range(ndim):
                x_test.append(np.random.uniform(low=x_min[i], high=x_max[i], size=args.num_test_points))
            x_test = (np.array(x_test)).T
        y_test = function.get_f(x_test)

        data['x_test'] = x_test.squeeze()  # squeeze for the 1D case
        data['y_test'] = y_test

    # print out a summary
    for i, (x, y) in enumerate(zip(x_trial, y_trial)):
        current_y = (np.array(y_trial))[0 : i + 1]
        print("Trial {}: x={}, y={}. Best is trial {} with value {}. Target is {}.".format(i, x, y, np.argmax(current_y), np.max(current_y), function_max), flush=True)


if __name__ == '__main__':
    main()