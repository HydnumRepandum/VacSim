# Can A Society of Generative Agents Simulate Human Behavior and Inform Public Health Policy? A Case Study on Vaccine Hesitancy.
## Introduction
This repo evaluates and analyzes a multi LLM agent framework, VacSim, for simulating health-related decision-making. Paper link: https://arxiv.org/abs/2503.09639

## Installation
1. Create and activate an environment in anaconda (recommend `python=3.10`)
2. `pip install -r requirements.txt`

### Running on Google Colab
To use VacSim on [Google Colab](https://colab.research.google.com/), clone the repository and install the dependencies in a notebook cell:

```python
!git clone https://github.com/<your-user>/VacSim.git
%cd VacSim
!pip install -r requirements.txt
```

Then set the appropriate API key(s) and run the driver script. For example, using OpenAI models:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # or set AZURE_OPENAI_API_KEY / ANTHROPIC_API_KEY
!python src/driver.py 1 --warmup_days 1 --run_days 1 --model_type gpt-4o-mini --news_path data/news/COVID-news-total-k=10000.pkl --network_str data/social_network-num=100-incl=neutral.pkl --profile_path data/profiles-num=100-incl=neutral.pkl --ports 7000 --temperature 0.7
```

### Using Gemma 3 270M Locally
VacSim can also run without OpenAI by loading the Gemma 3 270M model through Hugging Face. After installing the dependencies in `requirements.txt`, set `--model_type` to the Gemma checkpoint:

```python
!python src/driver.py 1 --warmup_days 1 --run_days 1 --model_type google/gemma-3-270m --news_path data/news/COVID-news-total-k=10000.pkl --network_str data/social_network-num=100-incl=neutral.pkl --profile_path data/profiles-num=100-incl=neutral.pkl --temperature 0.7
```

No API keys are required when running with Gemma; the model weights are downloaded and used locally via `transformers`.

## Codebase Overview
The main entry file of this repo is `driver.py`, which creates an `EvalSuite` object that conducts evaluations for the multi-agent system. Each `EvalSuite` (see implementations in `utils/eval_suite.py`) will analyze according to the `eval_mode` you input in:
- 0 -> conduct attitude tuning
- 1 -> incentive policy strength eval
- 2 -> community policy strength eval
- 3 -> mandate policy strength eval
- 4 -> news sanity eval
- 5 -> run simulation with different diseases (by replacing disease names)
- 6 -> compare simulation run under strong policies of each kind (incentive, community, mandate)

Each `EvalSuite` will create an `DataParallelEngine` or `AsyncDataParallelEngine` object, which will run parallel inferences on vLLM servers or make async API calls, depending on whether you run local or remote inferneces. `DataParallelEngine` or `AsyncDataParallelEngine` inherit `Engine` class, which specifies behaviors of agents and outlines their routines at each simulation time step.

### (If use open-source models) Starting vLLM Servers
- **(Recommended)** If you use bash script to run, you can see an example at `example_run_4_servers.sh` for running parallel servers on 4 GPUs
- If you use a single interative GPU, do something like: 
```
python -m vllm.entrypoints.openai.api_server \
      --model $model --guided-decoding-backend lm-format-enforcer --max-model-len 6144 \
      --tensor-parallel-size 1 --port $PORT 
```
where you input $model and $PORT. 

Example:
```
python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct --guided-decoding-backend lm-format-enforcer --max-model-len 6144 \
      --tensor-parallel-size 1 --port 49172 
```
- **Parallel**: If you use interactive GPUs on N parallel processes, request `N` GPUs and open `N` sessions. At each session, do the command above.

### Running Evals

Example command:

```
python src/driver.py 1 --warmup_days 5 --run_days 15 \
	--news_path data/news/COVID-news-total-k=10000.pkl \
	--network_str data/social_network-num=100-incl=neutral.pkl \
	--profile_path data/profiles-num=100-incl=neutral.pkl \
	--model_type meta-llama/Meta-Llama-3.1-8B-Instruct \
	--disease COVID-19 \
	--ports 49172 --temperature 0.7
```

This command runs policy strength eval (the incentive policy) for five seeds by default. If you want to supply a different list of seeds, add in `--seed_list` argument.

If you use multiple processes, then include all the ports, like:

```
python src/driver.py 1 --warmup_days 5 --run_days 15 \
	--news_path data/news/COVID-news-total-k=10000.pkl \
	--network_str data/social_network-num=100-incl=neutral.pkl \
	--profile_path data/profiles-num=100-incl=neutral.pkl \
	--model_type meta-llama/Meta-Llama-3.1-8B-Instruct \
	--disease COVID-19 \
	--ports 49172 55050 60050 60100 --temperature 0.7
```

### (Mandatory) Use OpenAI/Anthropic Models

If you use close-sourced models, we recommend to provide your API keys as environmental variables. 
If you use:
- **OpenAI models**: create an env variable called `OPENAI_API_KEY` and modify the `init_client` method in `src/engines/async_engine`.
- **OpenAI models hosted on Azure**: create env variables called `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` and specifiy the api_version the `init_client` method in `src/engines/async_engine`
- **Anthropic models**: create an env varibale called `ANTHROPIC_API_KEY`.

**Caveat**: Note that for some cloud providers, you need to specify the model token limit. For example, instead of `--model_type gpt-4o`, you need to do `--model_type gpt-4o-0513-50ktokenperminute`. Please be aware of this when you input.

### Further Instructions on Evals

The following lists commands and inputs for each kind of eval:
- **Attitude tuning (command 0)**: Do not provide `--temperature`, provide `--temperature_list` because it is tuning within a range. Example (on four ports).
```
python src/driver.py 0 --warmup_days 5 --run_days 15 \
	--news_path data/news/COVID-news-total-k=10000.pkl \
	--network_str data/social_network-num=100-incl=neutral.pkl \
	--profile_path data/profiles-num=100-incl=neutral.pkl \
	--model_type meta-llama/Meta-Llama-3.1-8B-Instruct \
	--disease COVID-19 \
	--ports 49172 55050 60050 60100 --temperature_list 0.1, 0.3, 0.5, 0.7, 1.0, 2.0
```

- **Policy Strength Eval (command 1,2,3)**: Provide `--temperature` (the selected temperature after attitude tuning). Example (on four ports).
 ```
python src/driver.py 1 --warmup_days 5 --run_days 15 \
	--news_path data/news/COVID-news-total-k=10000.pkl \
	--network_str data/social_network-num=100-incl=neutral.pkl \
	--profile_path data/profiles-num=100-incl=neutral.pkl \
	--model_type meta-llama/Meta-Llama-3.1-8B-Instruct \
	--disease COVID-19 \
	--ports 49172 55050 60050 60100 --temperature 0.7
```

- **News Eval (command 4)**: Provide a news_list (positive and negative news).
```
python src/driver.py 4 --warmup_days 5 --run_days 15 \
	--news_list data/news/COVID-news-positive-k=5000.pkl data/news/COVID-news-negative-k=5000.pkl \
	--network_str data/social_network-num=100-incl=neutral.pkl \
	--profile_path data/profiles-num=100-incl=neutral.pkl \
	--model_type meta-llama/Meta-Llama-3.1-8B-Instruct \
	--disease COVID-19 \
	--ports 49172 55050 60050 60100 --temperature 0.7
```

- **Policy Comparison (command 5)**. Need to specify a temperature, similar command to the Policy Strength Eval.
```
python src/driver.py 5 --warmup_days 5 --run_days 15 \
	--news_path data/news/COVID-news-total-k=10000.pkl \
	--network_str data/social_network-num=100-incl=neutral.pkl \
	--profile_path data/profiles-num=100-incl=neutral.pkl \
	--model_type meta-llama/Meta-Llama-3.1-8B-Instruct \
	--disease COVID-19 \
	--ports 49172 55050 60050 60100 --temperature 0.7
```
### Interpreting the Eval Results

The eval outputs will stack up your storage in the long term, as it contains the output of individual agents in the simulated population. If you need to save the output in a custom directory for data storage, specify the `--save_dir` argument, otherwise it will be created in the current directory.

The output directory contains the following files:
	- `results`: record the summary of all the simulation ran over a list of seeds, along with the individual run results.
	- `sim`: record the detailed info of every simulation run, including all agents output and a trajectory of an individual run.

### Customizing Policy Evals
You can create more evaluations by modifying the `eval` function in the `src/utils/eval_suite.py` file. You can set the policy, news, social network, and the demographic of agents provided to the engine by specifying corresponding parameters.