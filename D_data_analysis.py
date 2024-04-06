import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, OPTICS, Birch
from sklearn import metrics
import time as time
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd

#N.B. ONLY FOR PROJECTS ON DATA TYPES: USE THIS FUNCTION TO ENCODE CATEGORICAL VARIABLES BEFORE STANDARDIZATION
def encoding_categorical_variables(X):
    def encode(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=True)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return (res)

    categorical_columns=list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode(X,col)
    return X

@ignore_warnings(category=ConvergenceWarning)
def classification(X, y, classifier, seed):
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)
    clf = DecisionTreeClassifier()

    if classifier == "DecisionTree":
        clf = DecisionTreeClassifier()
    elif classifier == "LogisticRegression":
        clf = LogisticRegression()
    elif classifier == "KNN":
        clf = KNeighborsClassifier()
    elif classifier == "RandomForest":
        clf = RandomForestClassifier()
    elif classifier == "AdaBoost":
        clf = AdaBoostClassifier()
    elif classifier == "MLP":
        clf = MLPClassifier()

    print("Training for "+classifier+"...")

    start = time.time()

    model_fit = clf.fit(X, y)

    cv = ShuffleSplit(n_splits=8, test_size=0.3, random_state=seed)

    model_scores = cross_val_score(model_fit, X, y, cv=cv, scoring="f1_weighted")

    stop = time.time()
    speed = stop - start

    f1_mean = model_scores.mean()

    return {"mean_perf": f1_mean,
            "distance": distance_measurement(X, y, classifier, False, seed),
            "speed": speed}

@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=FutureWarning)
def regression(X, y, regression, seed):
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)
    regressor = Ridge()

    if regression == "LinearRegressor":
        regressor = Ridge()
    elif regression == "BayesianRidge":
        regressor = BayesianRidge()
    elif regression == "GPRegressor":
        regressor = GaussianProcessRegressor()
    elif regression == "SVMRegressor":
        regressor = LinearSVR()
    elif regression == "KNNRegressor":
        regressor = KNeighborsRegressor()
    elif regression == "MLPRegressor":
        regressor = MLPRegressor()

    print("Training for " + regression + "...")

    start = time.time()

    model_fit = regressor.fit(X, y)

    cv = ShuffleSplit(n_splits=8, test_size=0.3, random_state=seed)

    model_scores = cross_val_score(model_fit, X, y, cv=cv, scoring="neg_root_mean_squared_error")

    stop = time.time()
    speed = stop - start

    mse_mean = abs(model_scores.mean())

    return {"mean_perf": mse_mean,
            "distance": distance_measurement(X, y, regression, True, seed),
            "speed": speed}

def clustering(df, method, n_clusters, seed):
    def preprocess(df):
        df_log = abs(df)
        df_log = np.log1p(df_log)
        scaler = StandardScaler()
        scaler.fit(df_log)
        df_norm = scaler.transform(df_log)
        return df_norm

    df_norm = preprocess(df)

    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)

    if method == "KMeans":
        model.fit(df_norm)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(linkage="ward",
                                        n_clusters=n_clusters)  # linkage = "ward", "average", "complete", "single"
        model.fit(df_norm)
    elif method == "Spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=seed)
        model.fit(df_norm)
    elif method == "OPTICS":
        model = OPTICS(min_samples=round(len(df)/3))
        model.fit(df_norm)
    elif method == "BIRCH":
        model = Birch(n_clusters=n_clusters)
        model.fit(df_norm)

    print("Training for " + method + "...")

    start = time.time()

    silhouette = metrics.silhouette_score(df, model.labels_, metric="manhattan")

    stop = time.time()
    speed = stop - start

    return {"mean_perf": silhouette,
            "speed": speed}





def distance_measurement(X, y, method, regression, seed):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, mean_squared_error

    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)

    N = 8
    distances_train_test = np.zeros(N)
    for i in range(0, N):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed + i)

        model = DecisionTreeClassifier()

        if method == "DecisionTree":
            model = DecisionTreeClassifier()
        elif method == "LogisticRegression":
            model = LogisticRegression()
        elif method == "KNN":
            model = KNeighborsClassifier()
        elif method == "RandomForest":
            model = RandomForestClassifier()
        elif method == "AdaBoost":
            model = AdaBoostClassifier()
        elif method == "MLP":
            model = MLPClassifier()
        elif method == "LinearRegressor":
            model = Ridge()
        elif method == "BayesianRidge":
            model = BayesianRidge()
        elif method == "GPRegressor":
            model = GaussianProcessRegressor()
        elif method == "SVMRegressor":
            model = LinearSVR()
        elif method == "KNNRegressor":
            model = KNeighborsRegressor()
        elif method == "MLPRegressor":
            model = MLPRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_fit = model.predict(X_train)

        if regression:
            mse_pred = abs(mean_squared_error(y_test, y_pred, squared=False))
            mse_fit = abs(mean_squared_error(y_train, y_fit, squared=False))

            distances_train_test[i] = mse_pred - mse_fit
        else:
            weighted_f1_pred = f1_score(y_test, y_pred, average='weighted')
            weighted_f1_fit = f1_score(y_train, y_fit, average='weighted')

            distances_train_test[i] = weighted_f1_fit - weighted_f1_pred

    return distances_train_test.mean()
