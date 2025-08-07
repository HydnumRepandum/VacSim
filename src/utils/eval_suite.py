from tqdm import trange
import numpy as np
import os
from sandbox.policy import POLICY_REPO
from engines.configs import RunConfig, DataConfig, EngineConfig
from engines.multi_engine import DataParallelEngine
from engines.async_engine import AsyncDataParallelEngine

class EvalSuite:
    def __init__(self, args, file_dir, eval_mode):
        self.args = args
        self.file_dir = file_dir
        if not os.path.isdir(self.file_dir):
            os.mkdir(self.file_dir)
        self.eval_mode = eval_mode
        self.policies = []
        self.eval_data = {}
        self.create_engine()
    
    def init_eval_data(self):
        assert len(self.variables) > 0, "Please specify variables for this experiment"
        for var in self.variables:
            self.eval_data[var] = {
                "initial_hesitancies": [],
                "warmup_hesitancies": [],
                "average_decreases": [],
                "average_decreases_last_three": [],
                "monthly_data": [], # going to be a 2d list with a size of K seeds X N months
                "all_hesitancies": [], # going to be a 2d list with a size of K seeds X M weeks
            }
            
    
    def create_engine(self):
        data_config = DataConfig(
            news_path=self.args.news_path, 
            profile_str=self.args.profile_path,
            network_str=self.args.network_str,
            save_dir=os.path.join(self.args.save_dir, "sim")
        )
        engine_config = EngineConfig(
            model_type=self.args.model_type, 
            run_days=self.args.run_days, 
            warmup_days=self.args.warmup_days, 
            disease=self.args.disease, 
            ports=self.args.ports,
            alpha=self.args.alpha,
        )
        run_config = RunConfig(**data_config.__dict__, **engine_config.__dict__)

        if "anthropic" in self.args.model_type or "gpt" in self.args.model_type:
            engine = AsyncDataParallelEngine(**run_config.__dict__)
        else:
            engine = DataParallelEngine(**run_config.__dict__)
        self.engine = engine

    def add_eval_data(self, attitude_dist, var, seed=None):
        hesitancy_percentages = [att[0] for att in attitude_dist]
        self.eval_data[var]["all_hesitancies"].append(hesitancy_percentages)
        num_warmup_days = self.engine.warmup_days
        warmup_hesitancy = hesitancy_percentages[num_warmup_days]
        self.eval_data[var]["initial_hesitancies"].append(hesitancy_percentages[0])
        self.eval_data[var]["warmup_hesitancies"].append(warmup_hesitancy)
        self.eval_data[var]["average_decreases"].append(warmup_hesitancy - np.mean(hesitancy_percentages[num_warmup_days:]))
        last_three_days = min(3, self.engine.run_days - num_warmup_days)
        self.eval_data[var]["average_decreases_last_three"].append(warmup_hesitancy - np.mean(hesitancy_percentages[-1 * last_three_days:]))
        monthly_cutoff = [4, 8, 13, 17, 21, 26, 30]
        monthly_data = [hesitancy_percentages[num_warmup_days + i] for i in monthly_cutoff if num_warmup_days + i < len(hesitancy_percentages)]
        self.eval_data[var]["monthly_data"].append(monthly_data) 

    def add_and_reset(self, attitude_dist, var, seed=None):
        self.add_eval_data(attitude_dist, var=var, seed=seed)
        self.create_engine()

    def record_summary(self):
        with open(os.path.join(self.file_dir, "summary.tsv"), "w") as f:
            f.write("var_idx\tseed\tinitial_hesitancy\twarmup_hesitancy\taverage_decrease\taverage_decrease_last_three\tmonthly_data\n")
            for var_idx, var in enumerate(self.variables):
                f.write(f"{var}\n")
                for i in range(len(self.args.seed_list)):
                    f.write(f"{var_idx}\t{self.args.seed_list[i]}\t{self.eval_data[var]['initial_hesitancies'][i]:.2f}\t{self.eval_data[var]['warmup_hesitancies'][i]:.2f}\t{self.eval_data[var]['average_decreases'][i]:.2f}\t{self.eval_data[var]['average_decreases_last_three'][i]:.2f}\t{self.eval_data[var]['monthly_data'][i]}\n")
                f.write(f"Average\tN/A\t{np.mean(self.eval_data[var]['initial_hesitancies']):.2f}\t{np.mean(self.eval_data[var]['warmup_hesitancies']):.2f}\t{np.mean(self.eval_data[var]['average_decreases']):.2f}\t{np.mean(self.eval_data[var]['average_decreases_last_three']):.2f}\t{list(np.round(np.mean(self.eval_data[var]['monthly_data'], axis=0), 2))}\n")
            f.close()
        
    def eval(self):
        if self.eval_mode == 0: # attitude tuning
            assert self.args.temperature_list is not None and len(self.args.temperature_list) > 0, "Please specify temperature list for this experiment"
            self.variables = self.args.temperature_list
            self.init_eval_data()
            for temperature in self.variables:
                for i in trange(len(self.args.seed_list), desc=f"Running seed exp"):
                    seed = self.args.seed_list[i]
                    self.engine.set_seed(seed)
                    self.engine.set_temperature(temperature)
                    attitude_dist = self.engine.run_policy(policy=None, i=i)
                    self.add_and_reset(attitude_dist, var=temperature)

        elif self.eval_mode == 1 or self.eval_mode == 2 or self.eval_mode == 3: # incentive, community, mandate
            assert self.args.temperature is not None, "Please specify temperature for this experiment"
            policy_map = {1: "incentive", 2: "community", 3: "mandate"}
            policies = [policy for policy in POLICY_REPO if policy.cat == policy_map[self.eval_mode]]
            self.variables = [p.get_head() for p in policies]
            self.init_eval_data()
            for k in range(len(policies)):
                policy = policies[k]
                var = self.variables[k]
                for i in trange(len(self.args.seed_list), desc=f"Running seed exp"):
                    seed = self.args.seed_list[i]
                    self.engine.set_seed(seed)
                    self.engine.set_temperature(self.args.temperature)
                    attitude_dist = self.engine.run_policy(policy, i)
                    self.add_and_reset(attitude_dist, var=var)

        elif self.eval_mode == 4: # news sanity check
            assert self.args.news_list is not None and len(self.args.news_list) > 0, "Please specify news list for this experiment"
            self.variables = self.args.news_list
            self.init_eval_data()
            for news_path in self.variables:
                for i in trange(len(self.args.seed_list), desc=f"Running seed exp"):
                    seed = self.args.seed_list[i]
                    self.engine.set_seed(seed)
                    self.engine.set_temperature(self.args.temperature)
                    attitude_dist = self.engine.run_policy(policy=None, news_path=news_path, i=i)
                    self.add_and_reset(attitude_dist, var=news_path)
        
        elif self.eval_mode == 5: # policy compare
            assert self.args.temperature is not None, "Please specify temperature for this experiment"
            policies = [p for p in POLICY_REPO if p.strength == "strong"]
            self.variables = [p.get_head() for p in policies] # because objects have dynamic addresses and we will have key mistmatch if we set policies as the keys.
            self.init_eval_data()
            for k in range(len(policies)):
                policy = policies[k]
                var = self.variables[k]
                for i in trange(len(self.args.seed_list), desc=f"Running seed exp"):
                    seed = self.args.seed_list[i]
                    self.engine.set_seed(seed)
                    self.engine.set_temperature(self.args.temperature)
                    attitude_dist = self.engine.run_policy(policy, i)
                    self.add_and_reset(attitude_dist, var=var)
        self.record_summary()