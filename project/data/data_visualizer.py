import seaborn
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics

FIG_SIZE = (6, 6)


class DataVisualizer:
    def show_words_cloud(self, df):
        spam_words = ' '.join(list(df[df['label'] == 1]['text']))
        not_spam_words = ' '.join(list(df[df['label'] == 0]['text']))
        word_cloud_spam = WordCloud(width=1024, height=1024).generate(spam_words)
        word_cloud_not_spam = WordCloud(width=1024, height=1024).generate(not_spam_words)
        fig, subs = plt.subplots(1, 2)
        subs[0].imshow(word_cloud_spam)
        subs[1].imshow(word_cloud_not_spam)
        subs[0].title.set_text("Spam")
        subs[1].title.set_text("Not Spam")
        subs[0].axis("off")
        subs[1].axis("off")
        fig.suptitle("Word Cloud")
        plt.show()

    def show_count_plot(self, df):
        plt.figure(figsize=FIG_SIZE)
        sns.countplot(x=df["label"], data=df)
        plt.show()

    def show_TSNE(self, x, y):
        tsne_values = TSNE(random_state=1).fit_transform(x)
        df = pd.DataFrame(tsne_values)
        df["label"] = y.to_numpy()
        plt.figure(figsize=FIG_SIZE)
        plt.legend()
        ax = sns.scatterplot(x=0, y=1, hue="label", data=df)
        legend_labels, _ = ax.get_legend_handles_labels()
        ax.legend(legend_labels, ['Spam', 'Ham'])
        ax.set(xlabel="x")
        ax.set(ylabel="y")
        plt.show()

    def show_confusion_matrix(self, confusion_matrix, model_name):
        figure, ax = plt.subplots(figsize=FIG_SIZE)
        sns.heatmap(confusion_matrix, annot=True, linewidths=0.6, linecolor="green", fmt=".00f", ax=ax)
        plt.title("Confusion Matrix of {} model".format(model_name))
        plt.xlabel("Y prediction")
        plt.ylabel("Y true")
        plt.show()

    def show_train_val_loss(self, total_stats):
        df_total_stats = pd.DataFrame(data=total_stats)
        df_total_stats.set_index("epoch")
        plt.xticks(range(0, 10))
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Value")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_total_stats)
        seaborn.lineplot(data=df_total_stats["train_loss"], label="training loss")
        seaborn.lineplot(data=df_total_stats["val_loss"], label="validation loss")
        plt.show()

    def show_ruc_auc(self, test_y, predictions):
        fpr, tpr, thresholds = metrics.roc_curve(test_y, predictions)
        print('TPR is: {}'.format(tpr))
        print('FPR: is {}'.format(fpr))
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
