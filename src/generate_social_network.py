from utils.generate_utils import init_openai_client, request_GPT
import pickle
import argparse
from sandbox.agent import Agent
import networkx as nx
from tqdm import trange

def load_agents(profile_str):
    with open(profile_str, "rb") as f:
        # a list of dictionaries
        profiles = list(pickle.load(f))
    ids = list(range(len(profiles)))
    agents = [Agent(p) for p in profiles]
    for i in range(len(agents)):
        agents[i].id = ids[i]
    return agents



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_str", type=str, default="profiles/profiles-num=100-incl=neutral.pkl")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    client = init_openai_client(port=args.port)
    agents = load_agents(args.profile_str)
    edge_list = []

    for idx in trange(len(agents), desc="Total progress"):
        curr_agent = agents[idx]
        other_agents = agents[:idx] + agents[idx+1:]
        other_agents_str = [f"{i}. {a.get_profile_str()}" for i, a in enumerate(other_agents)]
        system_prompt = f"Pretend you are {curr_agent.get_profile_str()}. You are joining a social network. You will be provided a list of people in the network, where each person is described as 'ID. Gender\tAge:\tEducation:\tOccupation:\tPolitical belief:\tReligion: '. Which of these people will you become friends with? Provide a list of *YOUR* friends in the format ID, ID, ID, etc. Do not include any other text in your response. Do not include any people who are not listed below."
        user_prompt = f"Here are the people in the social network, separated by semicolon: {'; '.join(other_agents_str)}. Please ONLY provide a list of other people you would like to be friends with separated by commas. DO NOT PROVIDE OTHER TEXTS"
        num_try = 0; max_try = 10; generated = False
        
        while num_try < max_try and not generated:
            try:
                response = request_GPT(client, prompt=user_prompt, system_prompt=system_prompt, max_tokens=100, model=args.model)
                new_edges = [(idx,int(r)) for r in response.split(",") if int(r) != idx]
                edge_list.extend(new_edges)
                generated = True
            except:
                num_try += 1
                continue

    G = nx.DiGraph(edge_list)
    for u, v in G.edges():
        G[u][v]['weight'] = 3.0
    
    network_output = args.profile_str.replace("profiles", "social_network")
    with open(network_output, "wb") as f:
        pickle.dump(G, f)

        



