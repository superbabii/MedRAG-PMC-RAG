import os
import re
import json
import torch
import transformers
from transformers import AutoTokenizer
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *

class MedRAG:

    def __init__(self, llm_name="axiong/PMC_LLaMA_13B", rag=True, follow_up=False, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)

        self.templates = {
            "cot_system": general_cot_system,
            "cot_prompt": general_cot,
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag
        }

        self.max_length = 2048
        self.context_length = 1024
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name,
            cache_dir=self.cache_dir,
            legacy=False  # Switch to new behavior
        )
        
        # Load the model using bf16 for optimized memory usage
        self.model = transformers.LlamaForCausalLM.from_pretrained(
            self.llm_name, 
            cache_dir=self.cache_dir, 
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Automatically split across available devices
        )

        # self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')

        # Check if CUDA is available (for other use cases, but no need to move model manually)
        if torch.cuda.is_available():
            print("CUDA is available. Model is automatically moved by device_map.")
        else:
            print("CUDA not available. Using CPU.")

        # Ensure the tokenizer has a pad token if it doesn't already
        if self.tokenizer.pad_token is None:
            print("Tokenizer has no pad token, setting pad token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Print confirmation that the model has been loaded on devices
        print(f"Model automatically loaded on appropriate devices using `device_map`.")

    def _set_pad_token(self):
        """Ensure tokenizer has a valid pad_token_id."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def _generate_responses(self, prompt):
        # Extract the "content" field from the user's message
        user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
        
        # Simplified text generation
        inputs = self.tokenizer(
            user_message,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,  # Enable padding if batching
            truncation=True,  # Truncate to handle long prompts
            max_length=self.max_length
        )

        # No need to move inputs to the device manually with device_map="auto"
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs['input_ids'],
                max_length=self.max_length,  # Reduce length for faster responses
                do_sample=True,
                top_k=50,
                temperature=0.7,
                pad_token_id=self.model.config.pad_token_id
            )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def _retrieve_context(self, question, k, rrf_k, snippets=None, snippets_ids=None):
        """Retrieve relevant contexts or snippets based on the question."""
        if snippets is not None:
            retrieved_snippets = snippets[:k]
        elif snippets_ids is not None:
            if self.docExt is None:
                self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
            retrieved_snippets = self.docExt.extract(snippets_ids[:k])
        else:
            retrieved_snippets, _ = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
        
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, snip["title"], snip["content"]) for idx, snip in enumerate(retrieved_snippets)]
        return contexts if contexts else [""]

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir=None, snippets=None, snippets_ids=None):
        # Build the options text efficiently
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())]) if options else ''
        
        # Retrieve relevant contexts
        contexts = self._retrieve_context(question, k, rrf_k, snippets, snippets_ids) if self.rag else [""]

        answers = []
        for context in contexts:
            prompt = self.templates["medrag_prompt"].render(context=context, question=question, options=options_text)
            messages = [{"role": "system", "content": self.templates["medrag_system"]}, {"role": "user", "content": prompt}]
            answer = self._generate_responses(messages)
            answers.append(re.sub(r"\s+", " ", answer))

        # Save results if required
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)

        return answers[0] if len(answers) == 1 else answers