from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from data.data_visualizer import DataVisualizer


class ClassicModels:

    def create_model(self, name="SVC"):
        if name == "SVC":
            return SVC(kernel='sigmoid')
        elif name == "lrc":
            return LogisticRegression(solver='liblinear')
        elif name == "rfc":
            return RandomForestClassifier()
        elif name == "knc":
            return KNeighborsClassifier()

    def train(self, model, features, target):
        model.fit(features, target)

    def predict_model(self, model, features):
        return model.predict(features)

    def evaluate(self, prediction, test_y, model_name):
        print("Accuracy Score of {} model: {}".format(model_name, [accuracy_score(test_y, prediction)]))
        y_true = test_y
        confusion_matrix_value = confusion_matrix(y_true, prediction)
        data_visualizer = DataVisualizer()
        data_visualizer.show_confusion_matrix(confusion_matrix_value, model_name)
