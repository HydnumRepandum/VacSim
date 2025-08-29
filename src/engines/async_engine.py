import asyncio
import aiohttp
from engines.engine import Engine
import os
import time
from openai import AzureOpenAI, OpenAI
from anthropic import Anthropic
from utils.utils import clean_response, parse_lessons

class AsyncDataParallelEngine(Engine):
    def __init__(self, ports=None, batch_size=25, max_iter=5, delay=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.delay = delay  # Add delay to control the rate of requests
        self.max_iter = max_iter  # Max retries for parsing responses
        self.max_retries = 7  # Max retries for a single request
        self.init_client()

    def init_client(self):
        if "claude" in self.model_type:
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.azure_deployment = False
        elif "gpt" in self.model_type:
            if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
                api_version = "2023-05-15"
                self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=api_version,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                self.azure_deployment = True
            else:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.azure_deployment = False
        elif "gemma" in self.model_type:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_type)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            self.client = None
            self.azure_deployment = False
        else:
            raise ValueError("Unsupported model type")

    async def async_request_generate(self, session, prompt, max_tokens=80, gen_seed=None):
        """
        Asynchronously generate text from remote APIs or local Hugging Face models.
        """
        retry_attempts = 0

        while retry_attempts < self.max_retries:
            try:
                if "gemma" in self.model_type:
                    from torch import no_grad

                    def _sync_generate():
                        inputs = self.tokenizer.apply_chat_template(
                            prompt,
                            return_tensors="pt",
                            add_generation_prompt=True,
                        ).to(self.model.device)
                        with no_grad():
                            output = self.model.generate(
                                inputs,
                                max_new_tokens=max_tokens,
                                do_sample=True,
                                temperature=0.7,
                            )
                        generated = output[0][inputs.shape[-1]:]
                        return self.tokenizer.decode(generated, skip_special_tokens=True)

                    return await asyncio.to_thread(_sync_generate)

                if "claude" in self.model_type:
                    base_url = "https://api.anthropic.com/v1/messages"
                    headers = {
                        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01"
                    }
                    json_data = {
                        "model": self.model_type,
                        "system": prompt[0]['content'],
                        "messages": [prompt[1]],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                else:
                    if getattr(self, "azure_deployment", False):
                        base_url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{self.model_type}/chat/completions?api-version={self.client._api_version}"
                        headers = {
                            "api-key": self.client.api_key,
                            "Content-Type": "application/json"
                        }
                    else:
                        base_url = "https://api.openai.com/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {self.client.api_key}",
                            "Content-Type": "application/json"
                        }
                    json_data = {
                        "model": self.model_type,
                        "messages": prompt,
                        "max_tokens": max_tokens,
                        "seed": gen_seed,
                        "temperature": 0.7
                    }

                async with session.post(base_url, headers=headers, json=json_data) as response:
                    if response.status == 429:  # Rate limit error
                        retry_attempts += 1
                        retry_after = int(response.headers.get("Retry-After", 35))
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status >= 400:
                        raise Exception(f"HTTP error {response.status}: {await response.text()}")

                    completion = await response.json()
                    return completion['choices'][0]['message']['content'] if "claude" not in self.model_type else completion['content'][0]['text']
            except Exception as e:
                retry_attempts += 1
                if retry_attempts >= self.max_retries:
                    print(f"Request failed after {self.max_retries} retries: {e}")
                    return None

    async def async_request_generate_lesson(self, session, prompt,  max_tokens=80, day=None, gen_seed=None):
        """
        Generate lessons from prompts with retry logic for parsing.
        """
        num_iter = 0
        while num_iter < self.max_iter:
            response = await self.async_request_generate(session, prompt,  max_tokens, gen_seed)
            try:
                new_lessons = parse_lessons(response, day=day)
                return new_lessons
            except Exception as e:
                num_iter += 1
                print(f"Error in parsing lessons: {e}, retrying...")
                print(f"Response: {response}")

        print(f"Failed to parse lessons after {self.max_iter} attempts.")
        return []

    async def async_request_generate_attitude(self, session, prompt, max_tokens=80, day=None, gen_seed=None):
        """
        Generate attitudes from prompts with retry logic for parsing.
        """
        num_iter = 0
        success = False
        while num_iter < self.max_iter and not success:
            response = await self.async_request_generate(session, prompt, max_tokens, gen_seed)
            attitude_json, success = self.parse_attitude(response)
            num_iter += 1
            if not success:
                print(f"Error in parsing attitudes: {attitude_json}, retrying...")
        if not success:
            print(f"Failed to parse attitudes after {self.max_iter} attempts.")

        return attitude_json

    async def async_request_generate_actions(self, session, prompt, max_tokens=80, day=None, gen_seed=None):
        """
        Generate actions from prompts with retry logic for parsing.
        """
        num_iter = 0
        while num_iter < self.max_iter:
            response = await self.async_request_generate(session, prompt, max_tokens, gen_seed)
            try:
                action = clean_response(response)
                if len(action) > 2:
                    return action
            except Exception as e:
                num_iter += 1
                print(f"Error in parsing actions: {e}, retrying...")
                print(f"Response: {response}")

        print(f"Failed to parse actions after {self.max_iter} attempts.")
        return ""
    
    def chunkify(self, lst, n):
        """Split list into chunks of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    async def async_generate(self, max_tokens, day, f):
        """
        Asynchronous generate function to handle batches with retry logic.
        """
        print(f"Stage: {self.stage}, async generation started")
        start = time.time()
        func_call_dic = {
            "generate": self.async_request_generate,
            "generate_attitude": self.async_request_generate_attitude,
            "generate_actions": self.async_request_generate_actions,
            "generate_lessons": self.async_request_generate_lesson
        }

        results = []
        batch_inputs = list(self.chunkify(self.context, self.batch_size))
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, batch in enumerate(batch_inputs):
                generation_seeds = [int(s) for s in self.rng.integers(0, 10000, size=len(batch))]
                for prompt, seed in zip(batch, generation_seeds):
                    tasks.append(func_call_dic[f](session, prompt, max_tokens, day=day, gen_seed=seed))

            responses = await asyncio.gather(*tasks)
            results.extend(responses)

        end = time.time()
        print(f"Stage: {self.stage}, generation finished in {end - start:.2f} seconds")
        return results

    def generate(self, max_tokens, day, f):
        """
        Wrapper for async_generate to be called synchronously.
        """
        return asyncio.run(self.async_generate(max_tokens, day, f))
