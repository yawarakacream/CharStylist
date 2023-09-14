import json
import os
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image


class CSDataset(Dataset):
    cs_dataset_info: dict


class EtlcdbDataset(CSDataset):
    def __init__(self, char2idx, writer2idx, etlcdb_path, etlcdb_process_type, etlcdb_names, transforms):
        self.cs_dataset_info = {
            "etlcdb_process_type": etlcdb_process_type,
            "etlcdb_names": etlcdb_names,
        }
        self.transforms = transforms
        
        self.writer_groups = {} # {etlcdb_names: writer[]}
        self.items = [] # (image_path, word_idxes, writer_idx)
        
        for etlcdb_name in etlcdb_names:
            json_path = os.path.join(etlcdb_path, f"{etlcdb_name}.json")
            with open(json_path) as f:
                json_data = json.load(f)
            
            self.writer_groups[etlcdb_name] = []

            for item in json_data:
                relative_image_path = item["Path"] # ex) ETL4/5001/0x3042.png
                image_path = os.path.join(etlcdb_path, etlcdb_process_type, relative_image_path)
                
                char = item["Character"] # ex) "あ"
                if char not in char2idx:
                    char2idx[char] = len(char2idx)
                word = [char2idx[char]]
                
                serial_sheet_number = int(item["Serial Sheet Number"]) # ex) 5001
                writer = f"{etlcdb_name}_{serial_sheet_number}"
                if writer not in writer2idx:
                    writer2idx[writer] = len(writer2idx)
                    self.writer_groups[etlcdb_name].append(writer)
                
                # データ拡張した構造
                if etlcdb_process_type.startswith("+"):
                    image_dir = image_path[:-len(".png")]
                    relative_image_paths = list(os.listdir(image_dir))
                    relative_image_paths.sort()
                    for relative_image_path in relative_image_paths:
                        image_path = os.path.join(image_dir, relative_image_path)
                        self.items.append((image_path, word, writer))
                
                # 素の構造
                else:
                    self.items.append((image_path, word, writer2idx[writer]))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image_path, word_idxes, writer_idx = self.items[idx]
        
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        
        word_embedding = np.array(word_idxes, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()
        
        return image, word_embedding, writer_idx
