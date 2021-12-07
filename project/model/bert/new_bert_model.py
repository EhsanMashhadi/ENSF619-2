import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional

from sklearn.utils.class_weight import compute_class_weight
from transformers import AdamW, AutoModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

from data.data_visualizer import DataVisualizer
from model.bert.bert_util import BERT_MODEL, BertUtil
from model.bert.newbert_complex import NewBertComplex
from model.bert.newbert_simple import NewBertSimple
from utils.time_utils import format_time

device = torch.device("cuda")

# It is recommended to use these values for the learning rate 5e-5, 3e-5, 2e-5
OPTIMIZER_LR = 2e-5
EPOCHS = 10


class NewBertModel:

    def __init__(self, train_y, simple):
        self.simple = simple
        self.model = self.create_model()
        self.optimizer = self.optimizer()
        self.cross_entropy = self.entropy(train_y)
        self.data_visualizer = DataVisualizer()

    def create_model(self):
        bert = AutoModel.from_pretrained(BERT_MODEL)
        params = list(bert.named_parameters())
        print("BERT has {} Parameters".format(len(params)))
        print(bert)
        BertUtil.freeze_layers(bert)
        if self.simple:
            model = NewBertSimple(bert)
        else:
            model = NewBertComplex(bert)

        print("BERT has {} Parameters".format(len(params)))
        print(model)
        model = model.to(device=device)
        return model

    def optimizer(self, lr=OPTIMIZER_LR):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        return optimizer

    # compute the class weights for the labels to handle the class imbalance
    def entropy(self, train_y):
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
        weights = torch.tensor(weights, dtype=torch.float)
        weights = weights.to(device=device)
        cross_entropy = nn.NLLLoss(weight=weights)
        return cross_entropy

    def train(self, train_data_loader):
        print("Training:")
        self.model.train()
        total_loss = 0
        predictions = []
        for step, batch in enumerate(train_data_loader):
            batch = [b.to(device) for b in batch]
            sent_id, mask, labels = batch
            self.model.zero_grad()
            prediction = self.model(sent_id, mask)
            loss = self.cross_entropy(prediction, labels)
            total_loss += loss.item()
            if step % 5 == 0 and not step == 0:
                print('  Batch {} of {} Loss {}'.format(step, len(train_data_loader), loss.item()))
                print('  Total Loss {}'.format(total_loss))
            loss.backward()
            self.optimizer.step()
            prediction = prediction.detach().cpu().numpy()
            predictions.append(prediction)
        average_loss = total_loss / len(train_data_loader)
        predictions = np.concatenate(predictions, axis=0)
        print('Average Loss {}'.format(average_loss))
        return average_loss, predictions

    def evaluate(self, val_data_loader):
        print("Evaluating:")
        self.model.eval()
        total_loss = 0
        predictions = []

        for step, batch in enumerate(val_data_loader):
            batch = [b.to(device) for b in batch]
            sent_id, mask, labels = batch
            with torch.no_grad():
                prediction = self.model(sent_id, mask)
                loss = self.cross_entropy(prediction, labels)
                if step % 5 == 0 and not step == 0:
                    print('  Batch {} of {} Loss {}'.format(step, len(val_data_loader), loss.item()))
                    print('  Total Loss {}'.format(total_loss))
                total_loss += loss.item()
                prediction = prediction.detach().cpu().numpy()
                predictions.append(prediction)

        average_loss = total_loss / len(val_data_loader)
        print('Average Loss {}'.format(average_loss))
        predictions = np.concatenate(predictions, axis=0)
        return average_loss, predictions

    def run(self, train_data_loader, val_data_loader, epochs=EPOCHS):
        total_stats = []
        total_time_start = time.time()
        time0 = time.time()
        best_val_loss = float('inf')
        for epoch in range(epochs):
            print("Epoch {} from {} ---->".format(epoch, epochs))
            train_loss, _ = self.train(train_data_loader)
            time_train = time.time() - time0
            print("Train Time : {}".format(format_time(time_train)))
            time0 = time.time()
            val_loss, _ = self.evaluate(val_data_loader)
            time_val = time.time() - time0
            print("Validation Time {}".format(format_time(time_val)))
            if val_loss < best_val_loss:
                print("Saving best model to disk at epoch {}".format(epoch))
                torch.save(self.model.state_dict(), 'best_val_loss.pt')
                best_val_loss = val_loss
            print("Train Loss {}".format(train_loss))
            print("Val Loss {}".format(val_loss))
            total_stats.append(
                {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_time': time_train,
                    'val_time': time_val
                }
            )
        total_time_end = time.time() - total_time_start
        print("Total training time is {}".format(format_time(total_time_end)))
        self.data_visualizer.show_train_val_loss(total_stats)

    def predict(self, test_seq, test_mask, test_y):
        path = 'best_val_loss.pt'
        self.model.load_state_dict(torch.load(path))
        with torch.no_grad():
            predictions = self.model(test_seq.to(device), test_mask.to(device))
            predictions_numpy = predictions.detach().cpu().numpy()
            predictions_label = np.argmax(predictions_numpy, axis=1)
            print(classification_report(test_y, predictions_label))
            print(pd.crosstab(test_y, predictions_label))
            print("Accuracy is {}%".format(round(accuracy_score(test_y.detach().cpu(), predictions_label), 5) * 100))
            probabilist_values = functional.softmax(predictions, dim=1).cpu().numpy()
            print("ROC_AUC is {} ".format(round(roc_auc_score(test_y.detach().cpu(), probabilist_values[:, 1]), 5)))
            self.data_visualizer.show_ruc_auc(test_y.detach().cpu(), probabilist_values[:, 1])
