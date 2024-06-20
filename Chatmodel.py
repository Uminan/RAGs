from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

class ChatModel():
    # def __init__(self, model_path, device_id="cuda:6"):
    def __init__(self, **config):
        self.model = config.get("model", "llama3")
        # self.stop = self.config.get("stop", None)
        # self.stream = self.config.get("stream", True)
        self.model_path = config.get("model_path", "/mnt/zfs01/snowdar/pretrained/Sailor-7B-Chat")
        self.temperature = config.get("temperature", 0)
        self.max_tokens = config.get("max_tokens", 512)
        self.device = config.get("device_id", "cuda:6")
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 1)
        self.prompt = config.get("prompt", [])
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)

    def format(self, prompt):
        standard_format = self.tokenizer.apply_chat_template( 
        self.prompt, 
        tokenize=False, 
        add_generation_prompt=True)
        return standard_format

    def run_generation(self):
        formal_prompt = self.format(self.prompt)
        model_inputs = self.tokenizer([formal_prompt], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            eos_token_id=terminators,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        model_output = ""
        print("\nAssistant:\n", end='', flush=True)
        for new_text in streamer:
            model_output += new_text
            print(new_text, end='', flush=True)
        self.prompt.append({"role": "assistant", "content": model_output})

    def chat(self):
        
        print("\nAssistant: Greeting! I am an AI research assistant. How can I help you today?")
        try:
            while True:
                print("\nUser:")
                query = input("")
                if query.lower() == "exit":
                    break
                self.prompt.append({"role": "user", "content": query})
                self.run_generation()
        except KeyboardInterrupt:
            print("\n程序已终止.")

if __name__ == "__main__":
    prompt = [
            {"role": "system", "content": "You are an AI research assistant."},
            {"role": "assistant", "content": "Greeting! I am an AI research assistant. How can I help you today?"}
        ]
    config = {
        'prompt': prompt,
        'device_id': 'cuda:6',
        # 'model_path' :"/mnt/zfs01/snowdar/pretrained/Sailor-7B-Chat",
        'model_path':"/mnt/ceph01/snowdar/DS/pretrained/Meta-Llama-3-8B-Instruct",
        'max_tokens': 1024,
        'temperature': 0.3,
        'top_p': 0.95,
        'top_k': 1,
    }
    chat_model = ChatModel(**config)
    chat_model.chat()