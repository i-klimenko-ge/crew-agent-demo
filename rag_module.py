import torch
import json
import faiss
from transformers import AutoTokenizer, AutoModel
from typing import Dict

def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    # Mask padded tokens then average over sequence length.
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

def get_prompt_embedding(encoder, tokenizer, prompt_text, device: str) -> 'np.ndarray':
    prompt_prefix = 'query: '
    tokenized = tokenizer(
        prompt_prefix + prompt_text,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    # Move all tensors to the specified device.
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    outputs = encoder(tokenized['input_ids'])
    pooled = average_pool(outputs.last_hidden_state, tokenized['attention_mask'])
    # Convert to CPU numpy array for FAISS.
    return pooled.detach().to('cpu').numpy()

class SimpleSearch:
    _instance = None        

    def __new__(cls, encoder_path: str = 'intfloat/multilingual-e5-large', device: str = 'cuda', *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SimpleSearch, cls).__new__(cls)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(encoder_path)
            cls._instance.encoder = AutoModel.from_pretrained(encoder_path).to(device)
            cls._instance.device = device
        return cls._instance
    
    def __init__(self, index_path: str, texts_dict_path: str, *args, **kwargs):
        # Load the FAISS index.
        self.index = faiss.read_index(index_path)
        # Load the texts dictionary from file.
        with open(texts_dict_path, 'r') as fin:
            self.texts_dict = json.load(fin)
    
    def search(self, query: str, prefix: str = 'passage', add_context: bool = True, top_n: int = 5):
        # Compute the query embedding.
        prompt = get_prompt_embedding(self.encoder, self.tokenizer, query, self.device)
        distances, indices = self.index.search(prompt, top_n)
        results = []
        texts_count = len(self.texts_dict)
        
        for idx in indices[0]:
            parts = []
            # Add previous context if available.
            if add_context and idx > 0:
                prev_text = self.texts_dict.get(str(idx - 1), '')
                prev_text = prev_text.replace(prefix, '', 1).strip()
                if prev_text:
                    parts.append(prev_text)
            # Main text.
            main_text = self.texts_dict.get(str(idx), '')
            main_text = main_text.replace(prefix, '', 1).strip()
            parts.append(main_text)
            # Add next context if available.
            if add_context and idx < texts_count - 1:
                next_text = self.texts_dict.get(str(idx + 1), '')
                next_text = next_text.replace(prefix, '', 1).strip()
                if next_text:
                    parts.append(next_text)
            results.append(' '.join(parts))
        return results