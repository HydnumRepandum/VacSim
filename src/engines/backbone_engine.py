# This file contains the abstract backbone engine of the simulation
# It is useful for providing a concise overview of the simulation, i.e. see the run method
# It contains general methods such as message updates, saving, loading, etc.
import transformers
import torch
import numpy as np
import re
from functools import partial
import json
from datetime import datetime
from tqdm import trange
from sandbox.prompts import system_prompt
from sandbox.disease_model import NAME_TO_MODEL
# from sandbox.transmission_model import A_SIRV
import os
import pickle
from recommenders.tweet_recommender import TweetRecommender
from recommenders.news_recommender import NewsRecommender
from sandbox.agent import Agent
import logging

class BackboneEngine:
    def __init__(
        self, 
        profile_str=None,
        network_str=None,
        run_days = 30,
        warmup_days = 1,
        model_type = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        max_iter = 10,
        news_path = "data/news-k=400.pkl",
        policies_path = "data/test_3_policies.txt",
        save_dir = None, 
        disease = "FD-24",
        risk_data_path = None,
        seed = 42,
        temperature=1.0,
        alpha=0.3, # following bias
    ):
        # engine configurations
        self.temperature = temperature # attitude sampling temperature
        self.run_days = run_days
        self.warmup_days = warmup_days
        self.total_num_days = run_days + warmup_days
        if model_type == "gpt-4o":
            self.model_type = "gpt-4o-0513-50ktokenperminute"
        elif model_type == "gpt-4o-mini":
            self.model_type = "gpt-4o-mini-0718-100ktokenperminute" # TODO
        elif model_type == "anthropic":
            self.model_type = "claude-3-5-haiku-20241022"
        else:
            self.model_type = model_type
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_iter = max_iter
        self.disease = disease
        # breakpoint()
        self.risk_data_path = risk_data_path

        # run config
        self.context = None
        self.run_id = 0
        self.day = 1
        self.attitude_dist = []
        self.seed = seed
        self.set_seed()

        # data configurations
        self.profile_str = profile_str
        self.save_dir = os.path.join(os.environ['WEKA'], "run_cache") if save_dir == None else save_dir
        self.news_path = news_path
        self.network_str = network_str
        self.policies_path = policies_path
        
        # load data
        self.load_news()
        # self.load_policies()
        self.load_agents()
        self.tweet_recommender_alpha= alpha
        # initializing models
        self.tweet_recommender = TweetRecommender(alpha=alpha) 
        self.disease_model = NAME_TO_MODEL[self.disease](risk_data_path=self.risk_data_path, warmup_days=self.warmup_days)
        # self.transmission_model = A_SIRV(agents=self.agents, disease_model=name_to_model[self.disease], risk_data_path=risk_data_path, warmup_days=warmup_days)
    
    def _init_logger(self):
        """
        Initialize a logger for the engine.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers
        if not logger.handlers:
            # Create a file handler
            log_file = os.path.join(self.run_save_dir, 'engine.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Define a logging format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        logger.propagate = True

        return logger
    
    def set_seed(self, seed=None):
        if seed != None:
            self.seed = seed
        print(f"Setting seed: {self.seed}")
        self.rng = np.random.default_rng(self.seed)
        self.sampling_rng = np.random.default_rng(self.seed)
        transformers.set_seed(self.seed)
    
    def set_temperature(self, temperature):
        print(f"Setting temperature from {self.temperature} to {temperature}")
        self.temperature = float(temperature)
    
    def temperature_sampling(self, dist):
        res_dist = [1e-6 if v < 1e-6 else v for v in dist]
        # print("res_dist: ", res_dist)
        logits = np.log(res_dist)
        res_dist = np.exp(logits / abs(self.temperature-1e-6))
        total = sum(res_dist)
        res_dist = [v / total for v in res_dist]
        # print("res_dist after temperature: ", res_dist)
        return self.sample(res_dist)

    def sample(self, dist, min_p=0.0):
        # top-p sample
        new_dist = [v if v >= min_p else 0 for v in dist]
        # breakpoint()
        sum_dist = sum(new_dist)
        new_dist = [v / sum_dist for v in new_dist]
        attitude = int(self.sampling_rng.choice(np.arange(1, len(dist)+1), 1, replace=False, p=new_dist)[0])
        # round new_dist to 2 decimal places for better readability to LLMs
        new_dist = [round(v, 2) for v in new_dist]
        return attitude, new_dist

    def parse_distributions(self, response):
        try: 
            json_data = json.loads(response)
            if type(json_data) == list and type(json_data[0]) == float:
                return json_data, "No reasoning provided"
            if type(json_data) == list and type(json_data[0]) == dict:
                json_data = json_data[0]
            assert "attitude_dist" in json_data and "reasoning" in json_data, "Attitude distribution or reasoning not found"
            attitude_dist = json_data["attitude_dist"]
            assert type(attitude_dist) == list and len(attitude_dist) > 0 and type(attitude_dist[0]) == float, f"Attitude distribution is not a list of floats, element type: {type(attitude_dist[0])}, list type: {type(attitude_dist)}"
            reasoning = json_data["reasoning"]
            return attitude_dist, reasoning, True
        except ValueError as e:
            match = re.search(r'\[(.*?)\]', response) # edge case of only []
            if match:
                attitude_dist = match.group(1).split(",")
                try: 
                    attitude_dist = [float(v.strip()) for v in attitude_dist]
                except:
                    return [0.25, 0.25, 0.25, 0.25], "Parsing error", False
                return attitude_dist, "No reasoning provided", True 
            else:
                print("Error in parsing distributions: ", e)
                print("Original response: ", response)
                return [0.25, 0.25, 0.25, 0.25], "Parsing error", False

    def parse_attitude(self, response, temperature=1.0):
        try:
            orig_attitude_dist, reasoning, success = self.parse_distributions(response)
            # print("Attitude dist: ", attitude_dist)
            return {"reasoning": reasoning, "orig_attitude_dist": orig_attitude_dist}, success
            # return {"attitude": attitude, "reasoning": reasoning, "orig_attitude_dist": orig_attitude_dist, "attitude_dist": attitude_dist}, True
        except Exception as e:
            print("Error in parsing attitude: ", e)
            return {"reasoning": "I am not sure", "orig_attitude_dist": [0.25, 0.25, 0.25, 0.25]}, False
            
            # return {"attitude": 3, "reasoning": "I am not sure", "orig_attitude_dist": [0.25, 0.25, 0.25, 0.25], "attitude_dist": [0.25, 0.25, 0.25, 0.25]}, False
        
    def load_network(self):
        assert self.agents != None, "Agents must be loaded before loading the network"
        with open(self.network_str, "rb") as f:
            self.social_network = pickle.load(f)
            f.close()

        assert len(self.agents) == len(self.social_network), f"Number of agents must match the number of agents in the social network, but got: {len(self.agents)} and {len(self.social_network)}"
        adj_list = dict({k: list(dict(v).keys()) for k,v in dict(self.social_network.adj).items()})
        for i in range(len(self.agents)):
            self.agents[i].following = {v: 3 for v in adj_list[i]} # a dictionary of id to weight

    def load_agents(self):
        with open(self.profile_str, "rb") as f:
            # a list of dictionaries
            profiles = list(pickle.load(f))
        self.agents = [Agent(p) for p in profiles]
        self.num_agents = len(self.agents)
        ids = list(range(len(self.agents)))
        # load it for init_attitude
        for i in range(len(self.agents)):
            self.agents[i].id = ids[i]
        # load network 
        self.load_network()

    def load_news(self):
        with open(self.news_path, "rb") as f:
            self.news = pickle.load(f)
            f.close()
        for i in range(len(self.news)):
            self.news[i].text = self.news[i].text.replace("COVID-19", self.disease).replace("covid-19", self.disease).replace("Covid-19", self.disease).replace("COVID", self.disease).replace("covid", self.disease).replace("Covid", self.disease)
        self.news_recommender = NewsRecommender() 
        self.disease_broadcast_message = None
        self.recommended_news = None

    def reset(self):
        # reset run configs
        self.context = []
        self.day = 1
        self.attitude_dist = []

        # reload data
        self.load_agents()
        self.load_news()
        self.set_seed()
        
        # reload models
        self.tweet_recommender = TweetRecommender(alpha=self.tweet_recommender_alpha) 
        self.news_recommender = NewsRecommender()
        self.disease_model = NAME_TO_MODEL[self.disease](risk_data_path=self.risk_data_path, warmup_days=self.warmup_days)
        
    def reset_context(self):
        self.context = [system_prompt(self.disease, self.agents[i], self.day) for i in range(self.num_agents)]
        
    def add_prompt(self, new_prompts):
        self.reset_context()
        # different prompt for each agent
        if "gemma" in self.model_type:
            if type(new_prompts) == list:
                for k in range(self.num_agents):
                    # add reflections
                    system_prompt = self.context[k][0]['content']
                    prompt = system_prompt + "\n" + new_prompts[k] 
                    self.context[k] = [
                        {"role": "user", "content": prompt}
                    ]
            else:
                for k in range(self.num_agents):
                    system_prompt = self.context[k][0]['content']
                    prompt = system_prompt + "\n" + new_prompts
                    self.context[k] = [
                        {"role": "user", "content": prompt}
                    ]

        else:
            if type(new_prompts) == list:
                for k in range(self.num_agents):
                    # add reflections
                    self.context[k].append({
                        "role": "user", 
                        "content": new_prompts[k]}
                    )
        # same prompt
            else:
                for k in range(self.num_agents):
                    self.context[k].append({
                        "role": "user", 
                        "content": new_prompts}
                    )
    
    def add_all_lessons(self, new_lessons):
        for k in range(self.num_agents):
            self.agents[k].add_lessons(new_lessons[k])

    def save_agent(self, agent, k, cleaned_responses, agent_save_dir=None):
        agent_file_path = os.path.join(agent_save_dir, f"agent_id={agent.id}.tsv")
        
        if not os.path.exists(agent_file_path):
            with open(agent_file_path, "w") as f:
                f.write(f"Day\tStage\tResponse\tSys_Prompt\tUser_Prompt\tAll_Attitudes\tLessons\tReflections\tTweets\n")
        with open(agent_file_path, "a") as f:
            content1 = self.context[k][0]['content'].strip().replace("\n", " ").replace("\t", " ")
            content2 = self.context[k][1]['content'].strip().replace("\n", " ").replace("\t", " ") if len(self.context[k]) > 1 else ""
            response = cleaned_responses[k].strip().replace("\n", " ") if type(cleaned_responses[k]) == str else str(cleaned_responses[k]).strip().replace("\n", " ")
            tweets = [t.text.strip().replace("\n", " ") for t in agent.tweets]
            lessons = [str(l.text).strip().replace("\n", " ") for l in agent.lessons ]
            f.write(f"{self.day}\t{self.stage}\t{response}\t{content1}\t{content2}\t{agent.attitudes}\t{lessons}\t{agent.reflections}\t{tweets}\n")
            f.close()

    def save(self, cleaned_responses):
        file_path = os.path.join(self.run_save_dir, f"full_output.tsv")
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(f"Stage\tDay\tAttitude_Dist\tInput\tOutput\n")
        print("-" * 50)
        print(f"Saving to {file_path}")
        with open(file_path, "a") as f:
            f.write(f"{self.stage}\t{self.day}\t{self.attitude_dist}\t{self.context}\t{cleaned_responses}\n")
            f.close()
        print("-" * 50)
        print(f"Saving records for individual agents")
        agent_save_dir = os.path.join(self.run_save_dir, "agents")
        if not os.path.exists(agent_save_dir):
            os.makedirs(agent_save_dir)
        for k in range(len(self.agents)):
            agent = self.agents[k]
            self.save_agent(agent, k, cleaned_responses, agent_save_dir)
    
    def run(self, idx, policy, ablate_key=None):
        with open(os.path.join(self.run_save_dir, "run_config.json"), "w") as f:
            for k, v in self.__dict__.items():
                if k not in ["run_save_dir", "logger", "agents", "news", "disease_model", "tweet_recommender", "news_recommender", "recommended_news"]:
                    f.write(f"{k}: {v}\n")
        print("-"*50)
        policy_content = policy.content if policy != None else "None"
        print(f"**Running simulations of policy={policy_content}**")
        print(f"**Run ID: {self.run_id}**")
        functions_queue = [self.feed_news_data, self.feed_disease_broadcast, self.broadcast_news_and_policies, self.feed_tweets, self.prompt_actions, self.poll_attitude]
        functions_queue_no_tweet = functions_queue.copy()
        functions_queue_no_tweet.remove(self.feed_tweets)
        print("-"*50)
        print("**WARM-UP STARTED**")
        self.init_agents()
        for t in trange(self.warmup_days, desc="Warmup"):
            print(f"**WARM-UP DAY {t}**")
            execute_queue = functions_queue if t > 0 else functions_queue_no_tweet
            for func in execute_queue:
                func()
            self.day += 1
        print("**WARM-UP FINISHED**")

        # add policy
        self.broadcast_news_and_policies = partial(self.broadcast_news_and_policies, policy=policy)
        # replace the policy function in the queue
        functions_queue[2] = self.broadcast_news_and_policies

        # ablate_map = {
        #         7: [self.feed_news_data],
        #         8: [self.feed_disease_broadcast],
        #         9: [self.feed_tweets, self.prompt_actions],
        #         10: [self.feed_news_data, self.feed_disease_broadcast, self.feed_tweets, self.prompt_actions]
        #     }
        # ablate_functions = ablate_map[ablate_key] if ablate_key != None else None
        # if ablate_functions != None:
        #     for ablate_func in ablate_functions:
        #         # breakpoint()
        #         functions_queue.remove(ablate_func)

        for t in trange(self.warmup_days, self.total_num_days, desc=f"Running simulations of seed={idx}"):
            print(f"**DAY {t}**")
            for func in functions_queue:
                func()
            self.day += 1
        self.finish_simulation(self.run_id, policy_content)
        print(f"**Simulation of policy={policy_content} finished**")
        print("-"*50)
    

    def run_policy(self, policy, i, news_path=None, ablate_key=None):
        if news_path != None:
            self.news_path = news_path
            self.load_news()
        news_handle = self.news_path.split("/")[-1].replace(".pkl", "")
        self.curr_policy_head = policy.cat if policy != None else "None"
        self.run_id = f"{datetime.now().strftime('%y-%m-%d')}_{datetime.now().strftime('%H:%M:%S')}-news={news_handle}-policy={self.curr_policy_head}_num={i}_profiles={self.profile_str.split('/')[-1].replace('.pkl', '')}"
        self.run_save_dir = os.path.join(self.save_dir, f"{self.run_id}-model={self.model_type}-temp={self.temperature}-disease={self.disease}")
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        self.logger = self._init_logger()
        self.run(i, policy, ablate_key=ablate_key)
        return self.attitude_dist


    # TO-DO
    def validate_message(messages):
        return True



