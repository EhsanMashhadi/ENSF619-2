from data.data_loader import SampleDataLoader
from data.data_visualizer import DataVisualizer
from data.preprocessing import Preprocessing
from model.bert.bert_util import BertUtil
from model.bert.new_bert_model import NewBertModel
from model.classic.classic_models import ClassicModels


def show_data():
    data_loader = SampleDataLoader()
    data = data_loader.load(version=2)
    data_visualizer = DataVisualizer()
    data_visualizer.show_words_cloud(data)
    data_visualizer.show_count_plot(data)


def show_embedded_data(data):
    embedded_data = BertUtil.embed_data(data["text"][0:600])
    data_visualizer = DataVisualizer()
    data_visualizer.show_TSNE(embedded_data, data["label"][0:600])


def run_classical_models(data, model="knc"):
    preprocessing = Preprocessing()
    vectorized = preprocessing.word_to_vector(data)
    splitted_data = preprocessing.split_vectorized(vectorized, data["label"])
    model_builder = ClassicModels()
    svc = model_builder.create_model("knc")
    model_builder.train(svc, splitted_data["train_x"], splitted_data["train_y"])
    prediction = model_builder.predict_model(svc, splitted_data["test_x"])
    model_builder.evaluate(prediction, splitted_data["test_y"], model)


def tune_bert(splitted_data, simple):
    training = NewBertModel(splitted_data["train_y"], simple)

    data_loader = SampleDataLoader()

    train_seq, train_mask, train_y = data_loader.encode_data(splitted_data["train_x"], splitted_data["train_y"])
    train_loader = data_loader.create_loader(train_seq, train_mask, train_y)

    val_seq, val_mask, val_y = data_loader.encode_data(splitted_data["val_x"], splitted_data["val_y"])
    val_loader = data_loader.create_loader(val_seq, val_mask, val_y)

    test_seq, test_mask, test_y = data_loader.encode_data(splitted_data["test_x"], splitted_data["test_y"])
    training.run(train_loader, val_loader)
    training.predict(test_seq, test_mask, test_y)


# def show_AUC_ROC(splitted_data, simple):
#     training = NewBertModel(splitted_data["train_y"], simple)
#     data_loader = SampleDataLoader()
#     test_seq, test_mask, test_y = data_loader.encode_data(splitted_data["test_x"], splitted_data["test_y"])
#     training.predict(test_seq, test_mask, test_y)


def bert(simple):
    data_loader = SampleDataLoader()
    data = data_loader.load()
    preprocessing = Preprocessing()
    # max_len = preprocessing.count_tokens(data)
    # print("Max Len {}".format(max_len))
    splitted_data = preprocessing.split(data)
    # show_embedded_data(splitted_data)
    tune_bert(splitted_data, simple=simple)


if __name__ == '__main__':
    # run classical models
    # data_loader = SampleDataLoader()
    # data = data_loader.load()
    # run_classical_models(data, model="knc")
    # show_data()
    # tune bert
    # data_loader = SampleDataLoader()
    # data = data_loader.load()
    # preprocessing = Preprocessing()
    # splitted_data = preprocessing.split(data)
    # show_AUC_ROC(splitted_data, True)
    bert(False)

    # data_loader = SampleDataLoader()
    # data_loader.encode_data("Hello, this is ehsan")
    # preprocessing = Preprocessing()
    # splitted_data = preprocessing.split(data)
    # show_embedded_data(data)

# def dist(x, y):
#     return numpy.sqrt(numpy.sum((x - y) ** 2))
#
#
# a = numpy.array((1, 2, 3))
# b = numpy.array((5, 3, 4))
# dist_a_b = dist(a, b)
# print(dist_a_b)

# x1 = [[1], [2], [3]]
# y1 = [[5], [3], [4]]
#
# x2 = [[1], [10], [20]]
# y2 = [[50], [60], [70]]
# z2 = [[80], [90], [100]]
#
# d1 = sklearn.metrics.pairwise_distances(x1, y1, metric='euclidean')
# d1_scalar = numpy.sum(d1) / 2
#
# d21 = sklearn.metrics.pairwise_distances(x2, y2, metric='euclidean')
# d22 = sklearn.metrics.pairwise_distances(x2, z2, metric='euclidean')
# d23 = sklearn.metrics.pairwise_distances(y2, z2, metric='euclidean')
# d2_scalar = numpy.sum(numpy.array((d21, d22, d23))) / 3
#
#
# print(d1_scalar)
# print(d2_scalar)


# # train_seq, train_mask, train_y = data_loader.encode_data(splitted_data["train_x"], splitted_data["train_y"])


# check the deleted commands from the tutorial
# test on other dataset
# fix epoch and batch number
# remove break from the training and evaluation
# enhance the logging messages
# early stopping?
# check the model structure and find the best new structure for tunning

##Simple
#               precision    recall  f1-score   support
#
#            0       0.96      0.90      0.93        88
#            1       0.70      0.88      0.78        24
#
#     accuracy                           0.89       112
#    macro avg       0.83      0.89      0.85       112
# weighted avg       0.91      0.89      0.90       112
# AUC: 0.84


# Saving best model to disk at epoch 9
# Train Loss 0.5313337423971721
# Val Loss 0.503258191049099
# Total training time is 0:04:06


# complex

#               precision    recall  f1-score   support
#
#            0       0.99      0.92      0.95        95
#            1       0.67      0.94      0.78        17
#
#     accuracy                           0.92       112
#    macro avg       0.83      0.93      0.87       112
# weighted avg       0.94      0.92      0.92       112
#
# col_0   0   1
# row_0
# 0      87   8
# 1       1  16
# AUC: 0.9284829721362228


