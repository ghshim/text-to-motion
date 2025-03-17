import os
import sys
import re
import torch
import random
import numpy as np

base_path = os.path.abspath("/home/gahyeon/workspace/projects/elmoh/")
if base_path not in sys.path:
    sys.path.append(base_path)

# from src.data.dataset3 import get_sentence
from src.utils.utils import set_seed

def get_sentence(file_path, line_number=1):
    line_number = max(1, min(line_number, 4))
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i+1 == line_number:
                return line.split('#', 1)[0].strip()

    return line.split('#', 1)[0].strip() 


def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)

def sbert_encode(descriptions):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions)
   
    return embeddings

def bert_encode(descriptions, device='cpu'):
    from transformers import BertModel, BertTokenizer

    descriptions = [remove_punctuation(sentence) for sentence in descriptions]

    # Load tokenizer and bert   
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)

    # tokenize
    inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    attn_mask = inputs["attention_mask"] 

    # make embedding
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state * attn_mask.unsqueeze(-1)

    return embeddings


if __name__ == '__main__':
    from transformers import BertModel, BertTokenizer
    set_seed()

    line_numbers = [1, 2, 3, 4]
    device = torch.device('cuda:2')

    data_dir = '/home/gahyeon/workspace/data/humanml3d'
    descr_dir = os.path.join(data_dir, 'texts')
    embeddings_dir = os.path.join(data_dir, 'embeddings')
    
    descr_path_list = [os.path.join(descr_dir, description) for description in sorted(os.listdir(descr_dir)) ]
    descriptions = [get_sentence(path, line_number=random.choice(line_numbers)) for path in descr_path_list]
    descriptions = [remove_punctuation(sentence) for sentence in descriptions] # remove punctuation

    # Load bert tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)

    batch_size = 1

    for start_idx in range(0, len(descriptions), batch_size):
        end_idx = min(start_idx + batch_size, len(descriptions))
        batch_descriptions = descriptions[start_idx:end_idx]
        batch_descr_path_list = descr_path_list[start_idx:end_idx]

        # Tokenize
        inputs = tokenizer(batch_descriptions, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  
        attn_mask = inputs["attention_mask"]

        print(start_idx, batch_descriptions)
        # Make embedding
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state * attn_mask.unsqueeze(-1)

        assert len(batch_descr_path_list) == len(embeddings)

        os.makedirs(embeddings_dir, exist_ok=True)
        for e, path in zip(embeddings, batch_descr_path_list):
            filename = os.path.splitext(os.path.basename(path))[0]
            embedding = e.detach().cpu().numpy()
            print(embedding.shape)
            # np.save(os.path.join(embeddings_dir, filename), embedding)
        
        # print(f"Batch {start_idx // batch_size + 1} processed: {start_idx} to {end_idx}")
