import torch.nn as nn


class NewBertSimple(nn.Module):

    def __init__(self, bert1):
        super(NewBertSimple, self).__init__()
        self.bert = bert1
        self.dense = nn.Linear(768, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, result = self.bert(sent_id, mask, return_dict=False)
        result = self.dense(result)
        result = self.softmax(result)
        return result
