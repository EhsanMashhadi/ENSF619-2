import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

BERT_MODEL = "bert-base-uncased"


class BertUtil:
    @staticmethod
    def embed_data(data):
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
        tokens = tokenizer.batch_encode_plus(data, padding=True, return_token_type_ids=True)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']

        tokens_tensor = torch.tensor(ids)
        mask_tensor = torch.tensor(mask)
        model = BertModel.from_pretrained(BERT_MODEL)
        model.eval()
        with torch.no_grad():
            output = model(tokens_tensor, mask_tensor)
        encoded_data = pd.DataFrame(output[0][:, 0, :].numpy())
        return encoded_data

    @staticmethod
    def freeze_layers(bert):
        for param in bert.parameters():
            param.requires_grad = False
