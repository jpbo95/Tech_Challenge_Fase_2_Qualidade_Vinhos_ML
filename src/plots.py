import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def plot_figure(df, x_axis:str="", y_axis:str="", kind:str="", title:str="", x_label:str="", y_label:str=""):
    df.plot(
        x=x_axis,
        y=y_axis,
        kind=kind,
        legend=False,
        rot=0
    )

    plt.title(title)

    if len(x_label) > 0:
        plt.xlabel(x_label)

    if len(y_label) > 0:
        plt.ylabel(y_label)
    
    plt.show()


def plot_target_distribution(df, x_size:int=10, y_size:int=8, x_axis:str="", hue:str="", title:str="", x_label:str="", y_label:str="", xticklabel=[]):
    """
    Exibe a distribuição da variável alvo
    """

    plt.figure(figsize=(x_size, y_size))

    ax = sns.countplot(
        data=df,
        x=x_axis,
        palette=["#e74c3c", "#2ecc71"],
        hue=hue,
        legend=False
    )
    
    if len(xticklabel) > 0:
        ax.set_xticklabels(xticklabel)

    plt.title(title)
    
    if len(x_label) > 0:
        plt.xlabel(x_label)

    if len(y_label) > 0:
        plt.ylabel(y_label)

    plt.tight_layout()
    plt.show()


def plot_histograms(df):
    """
    Exibe histogramas de todas as variáveis numéricas,
    exceto a variável alvo.
    """

    columns = [c for c in df.columns if c != "qualidade"]

    df[columns].hist(
        figsize=(15, 12),
        bins=20,
        edgecolor="black"
    )

    plt.tight_layout()
    plt.show()


def plot_boxplot(df):

    plt.figure(figsize=(15,12))

    for i, col in enumerate(df.columns[:-1], 1):
        plt.subplot(4,3,i)
        sns.boxplot(y=df[col])

    plt.tight_layout()
    plt.show()


def plot_heatmap(df, x_size:int=12, y_size:int=12, title:str=""):

    plt.figure(figsize=(x_size, y_size))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",  linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_graph(df, x_size:int=12, y_size:int=12, kind:str="", title:str="", x_label:str="", y_label:str=""):
    plt.figure(figsize=(x_size, y_size))

    df.plot(kind=kind)
    plt.title(title)
    
    if len(x_label) > 0:
        plt.xlabel(x_label)

    if len(y_label) > 0:
        plt.ylabel(y_label)

    plt.show()

    
def plot_pairplots(df, columns=None, hue=None):
    if columns is None:
        columns = df.columns.tolist()

    cols = columns.copy()

    if hue is not None and hue not in cols:
        cols.append(hue)

    sns.pairplot(
        data=df[cols],
        vars=columns,
        hue=hue
    )


def plot_confusion_matrix(y_true, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", display_labels=["Baixa/Média\nQualidade", "Alta Qualidade"])

def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred)) 

