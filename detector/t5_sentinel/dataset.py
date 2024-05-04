import json
import torch
import torch.utils as utils
from transformers import T5TokenizerFast as Tokenizer
from detector.t5_sentinel.__init__ import config
from torch import Tensor
from typing import Tuple


class Dataset(utils.data.Dataset):
    '''
    Dataset for loading text from different large language models.

    Attributes:
        corpus (list[str]): The corpus of the dataset.
        label (list[str]): The labels of the dataset.
        tokenizer (Tokenizer): The tokenizer used.
    '''
    def __init__(self, partition: str, selectedDataset: Tuple[str] = ('Human', 'ChatGPT', 'PaLM', 'LLaMA')):
        super().__init__()
        
        self.corpus, self.label = [], []
        self.label_map = {'Human': 0, 'ChatGPT': 1, 'PaLM': 2, 'LLaMA': 3, 'GPT2': 4}  # Ensure these match your data labels
        filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
        for item in filteredDataset:
            with open(f'{item.root}/{partition}.jsonl', 'r') as f:
                for line in f:

                    if item.label == 'LLaMA':
                        words = json.loads(line)['text'].split()
                        continuation = words[75:]
                        if len(continuation) >= 42:
                            self.corpus.append(' '.join(continuation[:256]))
                            self.label.append(item.token)
                    else:
                        self.corpus.append(json.loads(line)['text'])
                        self.label.append(item.token)
                    
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(config.backbone.name, model_max_length=config.backbone.model_max_length)
        
    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.corpus[idx], self.label[idx]

    # def collate_fn(self, batch: Tuple[str, str]) -> Tuple[Tensor, Tensor, Tensor]:
    #     corpus, label = zip(*batch)
    #     corpus = self.tokenizer.batch_encode_plus(corpus, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
    #     label = self.tokenizer.batch_encode_plus(label, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
    #     print("Input ids are", label.input_ids)
    #     return corpus.input_ids, corpus.attention_mask, label.input_ids


        
    # def collate_fn(self, batch: Tuple[str, str]) -> Tuple[Tensor, Tensor, Tensor]:
    #     corpus, label = zip(*batch)
    #     corpus = self.tokenizer.batch_encode_plus(corpus, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        
    #     # Tokenizing labels individually
    #     tokenized_labels = [self.tokenizer.encode(label_text) for label_text in label]
        
    #     # Mapping label input_ids
    #     mapped_label_input_ids = []
    #     for label_ids in tokenized_labels:
    #         mapped_pair = [(32099, 0) if token_id == 32099 else (32098, 1) if token_id == 32098 else (32097, 1) if token_id == 32097 else (token_id, 2) for token_id in label_ids]
    #         mapped_label_input_ids.append([(token_id, mapped_value) for token_id, mapped_value in mapped_pair])
    
    #     # Extracting only token IDs and mapped values
    #     mapped_label_input_ids = [[token_id, mapped_value] for pair in mapped_label_input_ids for token_id, mapped_value in pair]
    #     print("Input ids are", mapped_label_input_ids)
    #     # Returning as a tuple
    #     return corpus.input_ids, corpus.attention_mask, tuple(mapped_label_input_ids)
    
        
    def collate_fn(self, batch: Tuple[str, str]) -> Tuple[Tensor, Tensor, Tensor]:
        corpus, label = zip(*batch)
        corpus = self.tokenizer.batch_encode_plus(corpus, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        
        # Tokenizing labels individually
        tokenized_labels = [self.tokenizer.encode(label_text) for label_text in label]

        
        # Mapping label input_ids
        mapped_label_input_ids = []
        for label_ids in tokenized_labels:

            mapped_ids = []
            if label_ids[0] == 32099:
                mapped_ids.append([32099, 0])  # Update the second value to 0 if the first value is 32099
            elif label_ids[0] == 32098:
                mapped_ids.append([32098, 1])  # Update the second value to 1 if the first value is 32098
            elif label_ids[0] == 32097:
                mapped_ids.append([32097, 2])  # Update the second value to 2 if the first value is 32097
            elif label_ids[0] == 32096:
                mapped_ids.append([32096, 3])  # Update the second value to 3 if the first value is 32096
            elif label_ids[0] == 32095:
                mapped_ids.append([32095, 4])  
            else:
                mapped_ids.append([label_ids[0], label_ids[1]])  # Keep other values unchanged
            mapped_label_input_ids.append(mapped_ids)
    
        # Flatten the list of lists
        mapped_label_input_ids_flat = [item for sublist in mapped_label_input_ids for item in sublist]

        # Returning as a list of lists
        return corpus.input_ids, corpus.attention_mask, torch.tensor(mapped_label_input_ids_flat)
