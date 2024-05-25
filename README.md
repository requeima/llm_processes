# Code for LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language
This repository contains the code to reproduce the experiments carried out in [LLM Processes: Numerical Predictive
Distributions Conditioned on Natural Language](https://arxiv.org/pdf/2405.12856).

The code has been authored by: John Bronskill, James Requeima, and Dami Choi.

## Dependencies
This code requires the following:
* python 3.9 or greater
* PyTorch 2.3.0 or greater
* transformers 4.41.0 or greater
* accelerate 0.30.1 or greater
* jsonargparse 4.28.0 or greater
* matplotlib 3.9.0 or greater
* optuna 3.6.1 or greater (only needed if you intend to run the black-box optimization experiments)

## LLM Support and GPU Requirements
We support a variety of LLMs through the Hugging Face transformer APIs. The code currently supports the following
LLMs:

| LLM Type     | URL    | GPU Memory Required (GB) |
| ---      | ---    |--------------------------| 
| phi-3-mini-128k-instruct | https://huggingface.co/microsoft/Phi-3-mini-128k-instruct | 8                     |
| llama-2-7B | https://huggingface.co/meta-llama/Llama-2-7b | 24                    |
| llama-2-70B | https://huggingface.co/meta-llama/Llama-2-70b | 160                   |
| llama-3-8B | https://huggingface.co/meta-llama/Meta-Llama-3-8B | 24                    |
| llama-3-70B | https://huggingface.co/meta-llama/Meta-Llama-3-70B | 160                   |
| mixtral-8x7B | https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 | 24                    |
| mixtral-8x7B-instruct | https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 | 160                   |

Adding a new LLM that supports the hugging face APIs is not difficult, just modify ```hf_ap.py```.

## Installation
1. Clone or download this repository.
2. Install the python libraries listed under dependencies.

## Running the code
* Change directory to the root directory of this repo.
* On linux run:
  * ```export PYTHONPATH=.```
* On Windows run:
  * ```set PYTHONPATH=.```

The main script is ```run_llm_process.py``` which supports many options that are defined in ```parse_args.py```.

From the root directory of the repo, run:
```python run_llm_process.py --llm_type <LLM Type> [additional options]```

Common options:

```--experiment_name <value>``` specifies a name that will be used to name any output or plot files,
default is ```test```.

```--output_dir <directory where output files are written>```, default is ```./output```.

```--plot_dir <directory where output plot files are written>```, default is ```./plots```.

```--num_samples <number of samples to take at each target location>```, default is ```50```.

```--autoregressive <True/False>```, if ```True```, run A-LLMP, if ```False```, run I-LLMP, default is ```False```.

```--batch_size <value>``` controls how many samples for each target point are processed at once. A higher value will
result in faster execution, but will consume more GPU memory. Lower this number if you get out of memory errors.
Default is ```5```.

## Reproducing the Experiments
### Prompt Engineering
The additional options are:

Data: ```--data_path <choose a file from the data/functions directory>```.
In the experiments we used ```sigmoid_10_seed_*.pkl```, ```square_20_seed_*.pkl```, and ```linear_cos_75_seed_*.pkl```,
where you would substitute a seed number for  the *.

Prompt Format: ```--x_prefix <value>```, ```--y_prefix <value>```, and ```--break_str <value>```

Prompt Order: ```--prompt_ordering <sequential/random/distance>```

Prompt y-Scaling: ```--y_min <value>``` and ```--y_max <value>```

Top-p and Temperature: ```--top_p <value>``` and ```--temperature <value>```

Autoregressive: ```--autoregressive True```

### 1D Synthetic Data
From the root directory of the repo, run:
```python ./experiments/run_functions_exp.py --llm_type <LLM Type> --function <beat/exp/gaussian_wave/linear/linear_cos/log/sigmoid/sinc/sine/square/x_times_sine/xsin>```

### Compare to LLMTime
From the root directory of the repo, run:
```python ./experiments/run_compare_exp.py --llm_type <LLM Type>```

### Fashion MNIST
From the root directory of the repo, run:
```python ./experiments/run_fashion_mnist_exp.py --llm_type <LLM Type>```

### Black-box Optimization
From the root directory of the repo, run:
```python ./experiments/run_black_box_opt_exp.py --llm_type <LLM Type> --experiment_name_prefix <see table> --function <see table> --max_generated_length <see table>  --num_cold_start_points <see table>```

| function | experiment_name_prefix | max_generated_length | num_cold_start_points |
|----------|-------------|----------------------|-----------------------|
| Sinusoidal | Sinusoidal | 7                    | 7                     |
| Gramacy | Gramacy | 8                    | 12                    |
| Branin | Branin | 7                    | 12                    |
| Bohachevsky | Bohachevsky | 11                   | 12                    |
| Goldstein | Goldstein | 12                   | 12                    |
| Hartmann3 | Hartmann3 | 7                    | 15                    |

### Simultaneous Temperature, Rainfall, and Wind Speed Regression
From the root directory of the repo, run:
```python run_llm_process.py --llm_type <LLM Type> --experiment_name weather_3 --data_path ./data/weather/weather_3.pkl --autoregressive True --num_decimal_places_y 1 --max_generated_length 20```

### In-context Learning Using Related Data Examples
From the root directory of the repo, run:
```python ./experiments/run_in_context.py --llm_type <LLM Type>```

### Conditioning LLMPs on Textual Information
#### Scenario-conditional Predictions 
From the root directory of the repo, run:
```python run_llm_process.py --llm_type <LLM Type> --data_path ./data/scenario/scenario_data_2_points.pkl --prefix <prompt to try> --autoregressive True --plot_trajectories 5 --forecast True```

#### Labelling Features Using Text
From the root directory of the repo, run:
```python ./experiments/run_housing_exp.py --llm_type <LLM Type>```


## Attributions
In the black-box optimization experiments, we use code from the [benchfunk](https://github.com/mwhoffman/benchfunk) repository (Copyright (c) 2014, the benchfunk authors).

The datasets in the ```data/functions``` directory are derived from the synthetic datasets in the [LLMTime](https://github.com/ngruver/llmtime)  repository (Copyright (c) 2023 Nate Gruver, Marc Finzi, Shikai Qiu).


## Contact
To ask questions or report issues, please open an issue on the issues tracker.

## Citation
If you use this code, please cite our paper:
```
@misc{requeima2024llm,
      title={LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language}, 
      author={James Requeima and John Bronskill and Dami Choi and Richard E. Turner and David Duvenaud},
      journal={arXiv preprint arXiv:2405.12856},
      year={2024},
}
```