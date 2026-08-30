import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def logistic_regression_model(X_train, y_train, random_state:int=42, class_weight:str="balanced") -> DecisionTreeClassifier:
    log_reg = LogisticRegression(random_state=random_state, class_weight=class_weight)
    log_reg.fit(X_train, y_train)
    return log_reg


def knn_model(X_train, y_train, n_neighbors:int=5) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def decision_tree_model(X_train, y_train, random_state:int=42) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(random_state=random_state)
    tree.fit(X_train, y_train)
    return tree


def random_forest_model(X_train, y_train, n_estimators:int=100, random_state:int=42) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def svm_model(X_train, y_train, kernel:str="rbf", random_state:int=42) -> SVC:
    svm = SVC(kernel=kernel, random_state=random_state)
    svm.fit(X_train, y_train)
    return svm
    