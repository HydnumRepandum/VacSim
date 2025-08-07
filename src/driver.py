import argparse
from datetime import datetime
import os
# from utils.evals import *
from utils.eval_suite import EvalSuite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=int, help="Experiment mode")
    parser.add_argument("--model_type", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--ports", type=int, default=7000, nargs="+")

    parser.add_argument("--warmup_days", type=int, default=0)
    parser.add_argument("--run_days", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.3, help="Following bias for the model")
    parser.add_argument('--alphas', type=float, default=None, nargs="+", help="List of alphas to use in the experiment")

    parser.add_argument("--disease", type=str, default="FD-24")
    
    parser.add_argument("--seed_list", type=int, default=[2621, 2749, 2909, 3083, 3259], nargs="+")

    parser.add_argument("--policy", type=int, default=None)
    
    parser.add_argument("--news_list", type=str, default=["data/news/COVID-news-positive-k=5000.pkl", "data/news/COVID-news-negative-k=5000.pkl"], nargs="+")
    parser.add_argument("--news_path", type=str, default="data/news/COVID-news-total-k=10000.pkl")

    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--save_dir", type=str, default="save_dir")
    
    parser.add_argument("--temperature_list", type=float, default=[1.0, 0.1, 0.5, 0.7, 1.5, 2.0], nargs="+")
    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--profile_path", type=str, default="data/profiles-num=100-incl=neutral.pkl")
    parser.add_argument("--network_str", type=str, default="data/social_network-num=100-incl=neutral.pkl")


    args = parser.parse_args()

    model_str = args.model_type.split("/")[-1]

    date = f"{datetime.now().strftime('%y-%m-%d')}_{datetime.now().strftime('%H:%M:%S')}"
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.isdir(os.path.join(args.save_dir, "results")):
        os.mkdir(os.path.join(args.save_dir, "results"))
    file_dir = os.path.join(args.save_dir, "results", f"{model_str}-time={date}-exp={args.exp}")
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    
    if args.exp != 0:
        assert args.temperature is not None, "Please specify temperature for this experiment"
    if args.alphas is None:
        eval_suite = EvalSuite(args, file_dir, args.exp)
        eval_suite.eval()
    else:
        for alpha in args.alphas:
            args.alpha = alpha
            eval_suite = EvalSuite(args, file_dir, args.exp)
            eval_suite.eval()
        

        
    
        
        
        

