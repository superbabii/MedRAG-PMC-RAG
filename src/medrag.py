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

        self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')

        self.model = transformers.LlamaForCausalLM.from_pretrained(self.llm_name, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self._set_pad_token()

    def _set_pad_token(self):
        """Ensure tokenizer has a valid pad_token_id."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.model.config.pad_token_id < 0:
            raise ValueError(f"Invalid pad_token_id: {self.model.config.pad_token_id}.")

    
    def _generate_responses(self, messages):
        # Extract the 'content' field from each message
        text_inputs = [msg["content"] for msg in messages]

        # Join the content of the messages into a single string for processing
        input_text = "\n".join(text_inputs)
        
        print("Input to tokenizer:", input_text)
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs['input_ids'],
                max_length=2048,
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
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())]) if options else ''
        contexts = self._retrieve_context(question, k, rrf_k, snippets, snippets_ids) if self.rag else []

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
    
    def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path=None, n_rounds=4, n_queries=3, qa_cache_path=None):
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())]) if options else ''
        context = ""
        qa_cache = []

        if qa_cache_path and os.path.exists(qa_cache_path):
            qa_cache = json.load(open(qa_cache_path))[:n_rounds]
            if qa_cache:
                context = qa_cache[-1]
            n_rounds -= len(qa_cache)

        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]
        for i in range(n_rounds):
            prompt = f"{context}\n\n{question}\n\n{options_text}" if context else f"{question}\n\n{options_text}"
            messages = [{"role": "system", "content": self.templates["i_medrag_system"]}, {"role": "user", "content": prompt}]
            response = self._generate_responses(messages)

            saved_messages.append({"role": "assistant", "content": response})
            if "## Answer" in response or "answer is" in response.lower():
                return response, saved_messages

            context += f"\n\n{response}"
            qa_cache.append(context)
            if qa_cache_path:
                with open(qa_cache_path, 'w') as f:
                    json.dump(qa_cache, f, indent=4)

        return response, saved_messages