import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer


# It is recommended to use batch size 16 or 32 for the fine-tuning the BERT
BATCH_SIZE = 32


class SampleDataLoader:

    def load(self, version=2):
        return pd.read_csv("spam_v{}.csv".format(version), encoding='latin-1')

    def encode_sample_text(self, text):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        encoded = tokenizer.batch_encode_plus(text, add_special_tokens=True, padding=True, return_token_type_ids=True)
        return encoded

    def encode_data(self, feature, label):
        tokens = self.encode_sample_text(feature)
        data_seq = torch.tensor(tokens["input_ids"])
        data_mask = torch.tensor(tokens["attention_mask"])
        data_y = torch.tensor(label.tolist())
        return data_seq, data_mask, data_y

    def create_loader(self, seq, mask, y, batch_size=BATCH_SIZE):
        dataset = TensorDataset(seq, mask, y)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
        return data_loader
