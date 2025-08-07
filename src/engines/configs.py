from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    news_path: str = "data/news/FD-total-k=1000.pkl"
    network_str: str = "social_network/social_network-num=100-incl=neutral.pkl"
    profile_str: str = "profiles/profiles-num=100-incl=neutral.pkl"
    save_dir: str = "run_cache/debug/"

@dataclass
class EngineConfig:
    model_type: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    disease: str = "FD-24"
    ports: List[int] = field(default_factory=list)
    run_days: int = 10
    warmup_days: int = 5
    max_iter: int = 10
    alpha: float = 0.3  # Following bias for the model
    temperature: float = 1.0
    risk_data_path: str = "data/data_table_for_weekly_deaths_and_weekly_%_of_ed_visits__the_united_states.csv"

@dataclass
class RunConfig(DataConfig, EngineConfig):
    seed: int = 42


    