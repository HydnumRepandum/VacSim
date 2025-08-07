from engines.engine import Engine
import multiprocessing as mp
import backoff
import openai
import os
import time
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
from utils.utils import clean_response, parse_lessons
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np

class DataParallelEngine(Engine):
    def __init__(self, ports=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ports = ports if ports and type(ports) == list else [80000]
        self.num_processes = len(self.ports)
        # Each process gets a unique randomizer
    
    def init_client(self, port=None):
        if "claude" in self.model_type:
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif "gpt" in self.model_type:
            api_version = "2023-09-01-preview"
            return AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=api_version,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        else:
            if port is None:
                raise Exception("Port is not provided")
            return OpenAI(base_url=f"http://0.0.0.0:{port}/v1")

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def request_generate(self, prompt, port, max_tokens=80, day=None, gen_seed=None):
        try:
            client = self.init_client(port)
            if "claude" in self.model_type:
                gen_func = client.messages.create
                args = {
                    "model": self.model_type,
                    "system": prompt[0]['content'],
                    "messages": [prompt[1]],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
            else:
                gen_func = client.chat.completions.create # Use the same randomizer, state preserved across calls
                # print(f"Generation Seed: {gen_seed}")
                args = {
                    "model": self.model_type,
                    "messages": prompt,
                    "seed": gen_seed,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
            completion = gen_func(**args)
            if "claude" in self.model_type:
                return completion.content[0].text
            return completion.choices[0].message.content
        except:
            raise Exception("Error in request_generate")

    def request_generate_lesson(self, prompt, port, max_tokens=80, day=None, gen_seed=None):
        num_iter = 0
        while num_iter < self.max_iter:
            response = self.request_generate(prompt, port, max_tokens=max_tokens, gen_seed=gen_seed)
            try:
                new_lessons = parse_lessons(response, day=day)
                return new_lessons
            except Exception as e:
                num_iter += 1
                print("Error in parsing lessons", e)
                print(response)
        
        raise Exception("Error in request_generate_lesson")

    def request_generate_attitude(self, prompt, port, max_tokens=80, day=None, gen_seed=None):
        num_iter = 0
        success = False
        while not success and num_iter < self.max_iter:
            response = self.request_generate(prompt, port, max_tokens=max_tokens, gen_seed=gen_seed)
            num_iter += 1
            attitude_json, success = self.parse_attitude(response)
        return attitude_json

    def request_generate_actions(self, prompt, port, max_tokens=80, day=None, gen_seed=None):
        action = ""
        while len(action) < 2:
            response = self.request_generate(prompt, port, max_tokens=max_tokens, gen_seed=gen_seed)
            action = clean_response(response)
        return action

    def chunkify(self, lst, n):
        """Split list into chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate(self, max_tokens, day, f):
        print(f"Stage: {self.stage}, generation started")
        start = time.time()
        func_call_dic = {
            "generate": self.request_generate,
            "generate_attitude": self.request_generate_attitude,
            "generate_actions": self.request_generate_actions,
            "generate_lessons": self.request_generate_lesson
        }
        results = []
        # Single port handling: 
        if self.num_processes == 1:
            for i, msg in enumerate(tqdm(self.context, desc="Generating")):
                gen_seed = int(self.rng.integers(0, 10000, size=1)[0])
                self.logger.info(f"Batch {i}: Seeds: {gen_seed}")
                res = func_call_dic[f](msg, self.ports[0], max_tokens, day, gen_seed)  # Use randomizer for process 0
                results.append(res)

        # Multi-process case
        else:
            batches = list(self.chunkify(self.context, self.num_processes))

            with mp.get_context('spawn').Pool(processes=self.num_processes) as pool:
                for i, batch in enumerate(batches):
                    # Ensure number of seeds matches the number of processes
                    generation_seeds = [int(s) for s in self.rng.integers(0, 10000, size=self.num_processes)]
                    
                    # Map ports to processes cyclically if processes > ports
                    ports = [self.ports[idx % len(self.ports)] for idx in range(self.num_processes)]
                    self.logger.info(f"Batch {i}: Seeds {generation_seeds} and Ports {ports}")
                    # Use ports and seeds mapped to the same processes consistently
                    res = pool.starmap(
                        func_call_dic[f],
                        [
                            (msg, port, max_tokens, day, generation_seeds[idx])
                            for idx, (msg, port) in enumerate(zip(batch, ports))
                        ]
                    )
                    results.extend(res)
            if isinstance(results[0], str):
                results = [clean_response(r) for r in results]
        end = time.time()
        self.logger.info(f"Stage: {self.stage}, generation finished in {end - start:.2f} seconds")
        print(f"Stage: {self.stage}, generation finished")
        print(f"Time taken for execution: {end - start}")
        return results

