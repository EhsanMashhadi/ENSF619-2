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


def bert(simple):
    data_loader = SampleDataLoader()
    data = data_loader.load()
    preprocessing = Preprocessing()
    splitted_data = preprocessing.split(data)
    tune_bert(splitted_data, simple=simple)


def count_max_tokens_bert():
    data_loader = SampleDataLoader()
    data = data_loader.load()
    preprocessing = Preprocessing()
    max_len = preprocessing.count_tokens(data)
    print("Max Token Length is {}".format(max_len))


if __name__ == '__main__':
    # count_max_tokens_bert()
    bert(False)
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
    # bert(False)

    # data_loader = SampleDataLoader()
    # data_loader.encode_data("Hello, this is ehsan")
    # preprocessing = Preprocessing()
    # splitted_data = preprocessing.split(data)
    # show_embedded_data(data)
