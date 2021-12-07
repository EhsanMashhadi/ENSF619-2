import torch.nn as nn


class NewBertComplex(nn.Module):

    def __init__(self, bert1):
        super(NewBertComplex, self).__init__()
        self.bert = bert1
        self.dense1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, result = self.bert(sent_id, mask, return_dict=False)
        result = self.dense1(result)
        result = self.relu(result)
        result = self.dropout(result)
        result = self.dense2(result)
        result = self.softmax(result)
        return result
