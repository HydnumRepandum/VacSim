# This file contains an abstract engine class that orchestrates the simulation
# It contains the concrete instantiation of some prompting methods and simulation flows
# However, it does not implement generate or generate_attitude methods, which are done by DataParallelEngine in multi_engine.py
# It is less on the high-level overview and more on the concrete prompt details (except generation)

from engines.backbone_engine import BackboneEngine
import json
from utils.utils import compile_enumerate
from utils.network_utils import homophily_corr
from collections import Counter
from utils.plot_utils import plot_attitudes
import os
import pickle
import networkx as nx
from sandbox.tweet import Tweet
from sandbox.prompts import *
from tqdm import trange

SHORT_TOKEN_LIMIT = 50
TWEET_TOKEN_LIMIT = 100
MED_TOKEN_LIMIT = 150
LONG_TOKEN_LIMIT = 250
FULL_TOKEN_LIMIT = 1000


class Engine(BackboneEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)      

    def init_agents(self):
        if self.day > 0:
            self.reset() # handle cases when the engine is reused
        
        self.stage = f"init_agents_day={self.day}"
        # breakpoint()
        profile_prompts = [init_prompt(self.disease) for i in range(self.num_agents)]
        self.add_prompt(profile_prompts)
        # breakpoint()
        json_data_list = self.generate(max_tokens=LONG_TOKEN_LIMIT, day=self.day, f="generate_attitude")
        attitudes = []
        for i in range(len(json_data_list)):
            json_data = json_data_list[i]
            # put sampling out of parallel processes
            attitude, new_dist = self.temperature_sampling(json_data["orig_attitude_dist"])
            attitudes.append(attitude)
            json_data_list[i]["attitude_dist"] = new_dist
            json_data_list[i]["attitude"] = attitude
        # update the message lists
        for j in range(self.num_agents):
            self.agents[j].attitudes.append(attitudes[j])
            self.agents[j].reasoning.append(json_data_list[j]["reasoning"])
            self.agents[j].attitude_dist.append(json_data_list[j]["attitude_dist"])
        self.update_attitude_dist(attitudes)  
        self.save(json_data_list)
        
    def update_attitude_dist(self, attitudes):
        num_against = len([att for att in attitudes if att < 3])
        num_swing = len([att for att in attitudes if att == 3])
        num_support = len([att for att in attitudes if att == 4])
        against_percentage = num_against / self.num_agents
        swing_percentage = num_swing / self.num_agents
        support_percentage = num_support / self.num_agents
        self.attitude_dist.append((against_percentage, swing_percentage, support_percentage))
        if not os.path.exists(os.path.join(self.run_save_dir,f"attitude_dist.tsv")):
            with open(os.path.join(self.run_save_dir,f"attitude_dist.tsv"), "w") as f:
                f.write("day\tagainst\tswing\tsupport\thomophily\thp1\thp2\thp3\thp4\n")
                f.close()
        for idx, att in enumerate(attitudes):
            self.social_network.nodes[idx]['attitude'] = att
            
        # plot_network(self.social_network, self.run_save_dir, self.day)
        homophily, same_one, same_two, same_three, same_four = homophily_corr(self.social_network)
        with open(os.path.join(self.run_save_dir,f"attitude_dist.tsv"), "a") as f:
            f.write(f"{self.day}\t{against_percentage:.2f}\t{swing_percentage:.2f}\t{support_percentage:.2f}\t{homophily:.2f}\t{same_one:.2f}\t{same_two:.2f}\t{same_three:.2f}\t{same_four:.2f}\n")
            f.close()

        plot_attitudes(self.attitude_dist, self.model_type, self.curr_policy_head, self.run_save_dir)
        
    def feed_news_data(self, num_news=3):
        search_space = num_news * num_news
        news_data = self.news[self.day * search_space: (self.day + 1) * search_space]
        recommendations = self.news_recommender.recommend(agents=self.agents, num_recommendations=num_news, news_data=news_data)
        all_news = []
        purities = []
        stances = []
        similarities = []
        for k in range(self.num_agents):
            # breakpoint()
            news_text, news_stance, news_sim = recommendations[k]
            news = compile_enumerate(news_text, header="News")
            binary_stance = [1 if s == "positive" else 0 for s in news_stance]
            purity = sum(binary_stance) / len(binary_stance) if sum(binary_stance) > len(binary_stance) / 2 else 1 - sum(binary_stance) / len(binary_stance)
            purities.append(purity)
            stances.append(sum(binary_stance) / num_news)
            similarities.append(sum(news_sim) / num_news)
            all_news.append(news)

        print(f"Average Purity: {sum(purities) / len(purities)}")
        print(f"Average Stance: {sum(stances) / len(stances)}")
        print(f"Average Similarity: {sum(similarities) / len(similarities)}")
        self.recommended_news = all_news

    def feed_disease_broadcast(self):
        disease_broadcast_message = disease_broadcast(self.disease, self.disease_model, self.day)
        for k in range(self.num_agents):
            self.agents[k].risk = self.disease_model.risks_categories[self.day]
        self.disease_broadcast_message = disease_broadcast_message
    
    def broadcast_news_and_policies(self, policy = None, num_news = 5):
        self.stage = f"feed_news_and_policies_day={self.day}"
        # either you have news or broadcast or both
        recommended_news = self.disease_broadcast_message if self.recommended_news == None else self.recommended_news
        if self.disease_broadcast_message != None:
            recommended_news = [self.disease_broadcast_message + recommended_news[i] for i in range(self.num_agents)]
        content = policy.content if policy != None else None
        prompts = [news_policies_prompt(self.disease, recommended_news[k], content, k=num_news) for k in range(self.num_agents)]
        for k in range(self.num_agents):
            self.agents[k].policy = policy        
        self.add_prompt(prompts)
        self.generate_and_save_lessons()
    
    def generate_and_save_lessons(self):
        new_lessons = self.generate(max_tokens=LONG_TOKEN_LIMIT, day=self.day, f="generate_lessons")
        self.add_all_lessons(new_lessons)
        save_data_format = [{
            "new_lessons": [[l.text, l.importance] for l in n_lessons]
        } for n_lessons in new_lessons]
        self.save(save_data_format)
        return new_lessons
        
    
    def feed_tweets(self, top_k=3, num_recommendations = 5):
        self.stage = f"feed_tweets_day={self.day}"
        recommendations = self.tweet_recommender.recommend(agents=self.agents, num_recommendations=num_recommendations) # e.g. 500 (num_agents) * 10 (num_tweets)
        print("Recommendations generated")
        prompts = [tweets_prompt(self.disease, [r[1] for r in recommendations if r[0]==k], top_k) for k in range(self.num_agents)]
        # print(f"Prompts generated, example: {prompts[0]}")
        self.add_prompt(prompts)
        self.stage = f"write_tweets_lesson_day={self.day}"
        self.generate_and_save_lessons()
    
    def prompt_actions(self):
        self.stage = f"prompt_actions_day={self.day}"
        self.add_prompt(action_prompt(self.disease))
        # breakpoint()
        actions = self.generate(max_tokens=TWEET_TOKEN_LIMIT, day=self.day, f="generate_actions")
        actions_tweets = [Tweet(text=actions[i], time=self.day, author_id=i) for i in range(len(actions))]
        for k in range(self.num_agents):
            self.agents[k].tweets.append(actions_tweets[k])
        self.save(actions)
        return actions

    def poll_attitude(self):
        self.add_prompt(attitude_prompt(self.disease))
        self.stage = f"poll_attitude_day={self.day}"
        json_data_list = self.generate(max_tokens=LONG_TOKEN_LIMIT, day=self.day, f="generate_attitude")
        attitudes = []
        for i in range(len(json_data_list)):
            json_data = json_data_list[i]
            attitude, new_dist = self.temperature_sampling(json_data["orig_attitude_dist"])
            attitudes.append(attitude)
            json_data_list[i]["attitude_dist"] = new_dist
            json_data_list[i]["attitude"] = attitude
        # update the message lists
        for j in range(self.num_agents):
            self.agents[j].attitudes.append(attitudes[j])
            self.agents[j].reasoning.append(json_data_list[j]["reasoning"])
        self.update_attitude_dist(attitudes)
        self.save(json_data_list)
    
    def finish_simulation(self, run_id, policy, top_k=5):
        # reject_reasons, reject_freqs = self.endturn_reflection(top_k)
        for k in range(self.num_agents):
            agent = self.agents[k]
            att_set = set(agent.attitudes)

        # save the simulation summary
        d = {
            "policy": policy,
            "vaccine_hesitancy_ratio": self.attitude_dist,
            "infection_info": {
                "risks_history": self.disease_model.risks,
                "risks_rate": self.disease_model.risks_change_rates,
            }
        }
        json_object = json.dumps(d, indent=4)
        path = os.path.join(self.run_save_dir, f"simulation_summary.json")
        with open(path, "w") as f:
            f.write(json_object)