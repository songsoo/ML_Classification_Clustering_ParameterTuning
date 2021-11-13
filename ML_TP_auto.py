import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from sklearn import preprocessing, mixture, metrics
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)


def getEncode(df, name, encoder):
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels


# onehot Encoding
def onehotEncode(df, name):
    le = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc = df[[name]]
    enc = le.fit_transform(enc).toarray()
    le.categories_[0] = le.categories_[0].astype(np.str)
    new = np.full((len(le.categories_[0]), 1), name + ": ")
    le.categories_[0] = np.core.defchararray.add(new, le.categories_[0])
    enc_df = pd.DataFrame(enc, columns=le.categories_[0][0])
    df.reset_index(drop=True,inplace=True)
    df = pd.concat([df, enc_df], axis=1)
    df.drop(columns=[name], inplace=True)
    return df


# label encoding
def labelEncode(df, name):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels


"""
:param X: feature values
:param numerical_columns: name of numerical columns (array of string)
:param categorical_columns: name of categorical columns (array of string)
:param scalers: array of scalers
:param encoders: array of encoders 
:param scaler_name: name of scalers (array of string)
:param encoder_name: name of encoders (array of string)
:return: 2d array that is scaled and encoded X 
"""


def get_various_encode_scale(X, numerical_columns, categorical_columns, scalers=None, encoders=None, scaler_name=None,
                             encoder_name=None):
    if categorical_columns is None:
        categorical_columns = []
    if numerical_columns is None:
        numerical_columns = []

    if len(categorical_columns) == 0:
        return get_various_scale(X, numerical_columns, scalers, scaler_name)
    if len(numerical_columns) == 0:
        return get_various_encode(X, categorical_columns, encoders, encoder_name)

    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
    if encoders is None:
        encoders = [preprocessing.LabelEncoder(), preprocessing.OneHotEncoder()]
        #encoders = [preprocessing.LabelEncoder()]

    after_scale_encode = [[0 for col in range(len(encoders))] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale_encode[i].pop()
        for encode in encoders:
            after_scale_encode[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                after_scale_encode[i][j][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
                j = j + 1
            i = i + 1

    for new in categorical_columns:
        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                    labelEncode(after_scale_encode[i][j], new)
                elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                    after_scale_encode[i][j] = onehotEncode(after_scale_encode[i][j], new)
                else:
                    getEncode(after_scale_encode[i][j], new, encoder)
                j = j + 1
            i = i + 1

    return after_scale_encode, scalers, encoders


"""
If there aren't categorical value, do this function
This function only scales given X
Return: 1d array of scaled X
"""


def get_various_scale(X, numerical_columns, scalers=None, scaler_name=None):
    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
        # scalers = [preprocessing.StandardScaler()]
    encoders = ["None"]

    after_scale = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale[i].pop()
        for encode in encoders:
            after_scale[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
        i = 0
        for scaler in scalers:
            after_scale[i][0][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
            i = i + 1

    return after_scale, scalers, ["None"]


"""
If there aren't numerical value, do this function
This function only encodes given X
Return: 1d array of encoded X
"""


def get_various_encode(X, categorical_columns, encoders=None, encoder_name=None):
    """
    Test scale/encoder sets
    """
    if encoders is None:
        # encoders = [preprocessing.LabelEncoder(),preprocessing.OneHotEncoder()]
        encoders = [preprocessing.LabelEncoder()]
    scalers = ["None"]

    after_encode = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_encode[i].pop()
        for encode in encoders:
            after_encode[i].append(X.copy())
        i = i + 1

    for new in categorical_columns:
        j = 0
        for encoder in encoders:
            if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                labelEncode(after_encode[0][j], new)
            elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                after_encode[0][j] = onehotEncode(after_encode[0][j], new)
            else:
                getEncode(after_encode[0][j], new, encoder)
            j = j + 1

    return after_encode, ["None"], encoders


"""
:param X: dataset
:param max_cluster: maximum number of clusters
:param n_inits: Number of time the k-means algorithm will be run with different centroid seeds.
:param max_iters: Maximum number of iterations of the k-means algorithm for a single run
:param tols: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
:param verboses: Verbosity mode.
:param random_state
"""


def kmeans(X, y, max_cluster=None, n_inits=None, max_iters=None, tols=None, verboses=None, random_state=None):

    if max_cluster is None:
        max_cluster = 7
    max_cluster = max_cluster + 1

    range_n_clusters = list(range(max_cluster))
    range_n_clusters.remove(0)
    range_n_clusters.remove(1)

    if n_inits is None:
        n_inits = [5,10,15, 20]
    if max_iters is None:
        max_iters = [300]
    if tols is None:
        tols = [1e-4]
    if verboses is None:
        verboses = [0]

    best_cluster = -1
    best_silhouette = -1
    best_n_inits = 0
    best_max_iters = 0
    best_tols = 0
    best_verboses = 0


    centerDF = pd.DataFrame

    for n_clusters in range_n_clusters:
        for n_init in n_inits:
            for max_iter in max_iters:
                for tol in tols:
                    for verbose in verboses:
                        print("number of clusters: ", n_clusters, "/ n_init:", n_init, "/ max_iter:", max_iter,
                              "/ tol:", tol, "/ verbose:", verbose)

                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.set_size_inches(18, 7)

                        ax1.set_xlim([-0.1, 1])
                        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                        clusterer = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol,
                                           verbose=verbose, random_state=random_state)
                        cluster_labels = clusterer.fit_predict(X)

                        silhouette_avg = silhouette_score(X, cluster_labels)
                        centers = clusterer.cluster_centers_

                        if best_silhouette < silhouette_avg:
                            best_silhouette = silhouette_avg
                            best_cluster = n_clusters
                            best_n_inits = n_init
                            best_max_iters = max_iter
                            best_tols = tol
                            best_verboses = verbose

                            sum = [0 for row in range(n_clusters)]
                            num = [0 for row in range(n_clusters)]

                            j = 0
                            for i in cluster_labels:
                                sum[i] = sum[i] + y[j]
                                num[i] = num[i] + 1
                                j = j + 1

                            for i in range(n_clusters):
                                sum[i] = sum[i] / num[i]
                            centerDF = pd.DataFrame(centers)
                            # centerDF.loc[:, 'Mean House Value'] = sum

                        print("The average silhouette_score is :", silhouette_avg)

                        sample_silhouette_values = silhouette_samples(X, cluster_labels)

                        y_lower = 10
                        for i in range(n_clusters):
                            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                            ith_cluster_silhouette_values.sort()

                            size_cluster_i = ith_cluster_silhouette_values.shape[0]
                            y_upper = y_lower + size_cluster_i

                            color = cm.nipy_spectral(float(i) / n_clusters)
                            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                              0, ith_cluster_silhouette_values,
                                              facecolor=color, edgecolor=color, alpha=0.7)

                            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                            y_lower = y_upper + 10  # 10 for the 0 samples

                        ax1.set_title("Silouette Plot")
                        ax1.set_xlabel("Silhouette coefficient")
                        ax1.set_ylabel("Cluster label")

                        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                        ax1.set_yticks([])  # Clear the yaxis labels / ticks
                        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                    c=colors, edgecolor='k')

                        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

                        for i, c in enumerate(centers):
                            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

                        ax2.set_title("Cluster")
                        ax2.set_xlabel("1st Column")
                        ax2.set_ylabel("2nd Column")

                        plt.suptitle((
                                     "Kmeans, N clusters: ", n_clusters, " n_inits: ", n_inits, " max_iter: ", max_iter,
                                     " tol: ", tol, " verbose: ", verbose), fontsize=14, fontweight='bold')


    plt.show()
    df = centerDF.copy()
    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ n_init:", best_n_inits, "/ max_iter:", best_max_iters, "/ tol:", best_tols, "/ verbose:",
          best_verboses, "\n")
    param = 'Best params_/ best cluster: ' + str(best_cluster) + '/ n_init: ' + str(
        best_n_inits) + ' / max_iter: ' + str(best_max_iters) + '/ tol: ' + str(best_tols) + '/ verbose: ' + str(
        best_verboses)
    return best_silhouette, param, list(X.columns)


"""
:param X: dataset
:param max_cluster: maximum number of clusters
:param covariance_types: String describing the type of covariance parameters to use. 
:param tols: The convergence threshold.
:param max_iters: The number of initializations to perform.
:param n_inits: The number of initializations to perform.
:param random_state
"""


def GMM(X, y, max_cluster=None, covariance_types=None, tols=None, max_iters=None, n_inits=None, random_state=None):
    if max_cluster is None:
        max_cluster = 6
    max_cluster = max_cluster + 1

    if covariance_types is None:
        covariance_types = ['full']
    if tols is None:
        tols = [1e-3]
    if max_iters is None:
        max_iters = [50, 100, 150]
    if n_inits is None:
        n_inits = [1, 3, 5, 7, 10]

    range_n_clusters = list(range(max_cluster))
    range_n_clusters.remove(0)
    range_n_clusters.remove(1)

    best_cluster = -1
    best_silhouette = -1
    best_covariance_type = ''
    best_tol = 0
    best_max_iter = 0
    best_n_init = 0

    centerDF = pd.DataFrame

    for n_clusters in range_n_clusters:
        for covariance_type in covariance_types:
            for tol in tols:
                for max_iter in max_iters:
                    for n_init in n_inits:
                        print("number of clusters: ", n_clusters, "/ covariance type:", covariance_type, "/ n_init:",
                              n_init, "/ max_iter:", max_iter,
                              "/ tol:", tol)

                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.set_size_inches(18, 7)

                        ax1.set_xlim([-0.1, 1])
                        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                        clusterer = mixture.GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                                                            tol=tol, max_iter=max_iter, n_init=n_init)
                        clusterer.fit(X)
                        cluster_labels = clusterer.predict(X)

                        silhouette_avg = silhouette_score(X, cluster_labels)
                        print("The average silhouette_score is :", silhouette_avg)

                        # Labeling the clusters
                        centers = clusterer.means_

                        if best_silhouette < silhouette_avg:
                            best_silhouette = silhouette_avg
                            best_cluster = n_clusters
                            best_covariance_type = covariance_type
                            best_tol = tol
                            best_max_iter = max_iter
                            best_n_init = n_init

                            sum = [0 for row in range(n_clusters)]
                            num = [0 for row in range(n_clusters)]

                            j = 0
                            for i in cluster_labels:
                                sum[i] = sum[i] + y[j]
                                num[i] = num[i] + 1
                                j = j + 1

                            for i in range(n_clusters):
                                sum[i] = sum[i] / num[i]
                            centerDF = pd.DataFrame(centers)
                            # centerDF.loc[:, 'Mean House Value'] = sum

                        sample_silhouette_values = silhouette_samples(X, cluster_labels)

                        y_lower = 10
                        for i in range(n_clusters):
                            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                            ith_cluster_silhouette_values.sort()

                            size_cluster_i = ith_cluster_silhouette_values.shape[0]
                            y_upper = y_lower + size_cluster_i

                            color = cm.nipy_spectral(float(i) / n_clusters)
                            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                                              facecolor=color, edgecolor=color, alpha=0.7)

                            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                            y_lower = y_upper + 10  # 10 for the 0 samples

                        ax1.set_title("Silouette Plot")
                        ax1.set_xlabel("Silhouette coefficient")
                        ax1.set_ylabel("Cluster label")

                        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                        ax1.set_yticks([])  # Clear the yaxis labels / ticks
                        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                        # 2nd Plot showing the actual clusters formed
                        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                    c=colors, edgecolor='k')

                        # Draw white circles at cluster centers
                        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                    c="white", alpha=1, s=200, edgecolor='k')

                        for i, c in enumerate(centers):
                            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                        s=50, edgecolor='k')

                        ax2.set_title("Cluster")
                        ax2.set_xlabel("1st Column")
                        ax2.set_ylabel("2nd Column")

                        plt.suptitle(("GMM n_clusters: ", n_clusters, " "),
                                     fontsize=14, fontweight='bold')

    plt.show()

    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ covariance_types:", best_covariance_type, "/ max_iter:", best_max_iter, "/ tol:", best_tol,
          "/ n_init:",
          best_n_init, "\n")
    param = "Best params_/ cluster: " + str(
        best_cluster) + "/ covariance_types:" + best_covariance_type + "/ max_iter:" + str(
        best_max_iter) + "/ tol:" + str(best_tol) + "/ n_init:" + str(best_n_init)
    return best_silhouette, param, list(X.columns)


"""
:param X: datasets
:param epsS: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
:param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
:param metrics: The metric to use when calculating distance between instances in a feature array
:param algorithms: The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors
:param leaf_sizes
 """


def DBSCANs(X, y, epsS=None, min_samples=None, metrics=None, algorithms=None, leaf_sizes=None):
    if epsS is None:
        epsS = [0.1, 0.5, 0.7, 1.0]
    if min_samples is None:
        min_samples = [3, 4, 5]
    if metrics is None:
        metrics = ['euclidean']
    if algorithms is None:
        algorithms = ['auto']
    if leaf_sizes is None:
        leaf_sizes = [20, 30, 40]

    best_silhouette = -1
    best_cluster = -1
    best_eps = 0
    best_min_sample = 0
    best_metric = ''
    best_algorithm = ''
    best_leaf_size = 0

    centerDF = pd.DataFrame()

    for eps in epsS:
        for min_sample in min_samples:
            for metric in metrics:
                for algorithm in algorithms:
                    for leaf_size in leaf_sizes:
                        np.set_printoptions(threshold=100000, linewidth=np.inf)

                        clusterer = DBSCAN(eps=eps, min_samples=min_sample, metric=metric, algorithm=algorithm,
                                           leaf_size=leaf_size).fit(X)
                        cluster_labels = clusterer.labels_

                        n_clusters = len(set(clusterer.labels_))

                        if n_clusters < 2:
                            continue

                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.set_size_inches(18, 7)

                        unique_set = set(clusterer.labels_)
                        unique_list = list(unique_set)
                        if unique_list.count(-1):
                            unique_list.remove(-1)

                        a = np.array([[0 for col in range(len(X.iloc[0, :]))] for row in range(len(set(unique_list)))],
                                     dtype=np.float64)
                        num = np.array([0 for row in range(len(set(unique_list)))])

                        i = 0
                        for cluster in cluster_labels:
                            if (cluster != -1):
                                a[cluster] = a[cluster] + X.iloc[i, :]
                                num[cluster] = num[cluster] + 1
                            i = i + 1

                        i = 0

                        for cluster in unique_list:
                            a[cluster] = a[cluster] / num[cluster]

                        ax1.set_xlim([-0.1, 1])
                        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
                        # print(cluster_labels)
                        silhouette_avg = silhouette_score(X, cluster_labels)
                        print("number of clusters: ", n_clusters, "/ eps:", eps, "/ min_sample:", min_sample,
                              "/ metric:", metric, "/ algorithm:", algorithm, "/ leaf_size:", leaf_size)
                        print("The average silhouette_score is :", silhouette_avg)

                        centers = np.array(a)

                        if best_silhouette < silhouette_avg:
                            best_silhouette = silhouette_avg
                            best_cluster = n_clusters
                            best_eps = eps
                            best_metric = metric
                            best_algorithm = algorithm
                            best_leaf_size = leaf_size
                            best_min_sample = min_sample

                            sum = [0 for row in range(n_clusters)]
                            num = [0 for row in range(n_clusters)]
                            j = 0
                            for i in cluster_labels:
                                if i >= 0:
                                    sum[i] = sum[i] + y[j]
                                    num[i] = num[i] + 1
                                    j = j + 1

                            for i in range(n_clusters):
                                if num[i] != 0:
                                    sum[i] = sum[i] / num[i]
                            centerDF = pd.DataFrame(centers)
                            if len(sum) != len(centerDF):
                                sum.pop()
                            # centerDF.loc[:, 'Mean House Value'] = sum

                        sample_silhouette_values = silhouette_samples(X, cluster_labels)

                        y_lower = 10
                        for i in range(n_clusters):
                            ith_cluster_silhouette_values = \
                                sample_silhouette_values[cluster_labels == i]

                            ith_cluster_silhouette_values.sort()

                            size_cluster_i = ith_cluster_silhouette_values.shape[0]
                            y_upper = y_lower + size_cluster_i

                            color = cm.nipy_spectral(float(i) / n_clusters)
                            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                                              facecolor=color,
                                              edgecolor=color, alpha=0.7)

                            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                            y_lower = y_upper + 10  # 10 for the 0 samples

                        ax1.set_title("Silouette Plot")
                        ax1.set_xlabel("Silhouette coefficient")
                        ax1.set_ylabel("Cluster label")

                        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                        ax1.set_yticks([])  # Clear the yaxis labels / ticks
                        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                    c=colors, edgecolor='k')

                        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                    c="white", alpha=1, s=200, edgecolor='k')

                        for i, c in enumerate(centers):
                            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                        s=50, edgecolor='k')

                        ax2.set_title("Cluster")
                        ax2.set_xlabel("1st Column")
                        ax2.set_ylabel("2nd Column")

                        plt.suptitle(("Silhouette analysis for DBSCAN clustering on sample data "
                                      "with n_clusters = %d" % n_clusters),
                                     fontsize=14, fontweight='bold')

    print("---------------------\n", centerDF)

    plt.show()
    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ eps:", best_eps, "/ min_sample:", best_min_sample, "/ metric:", best_metric, "/ algorithm:",
          best_algorithm, "/ leaf_size:", best_leaf_size, "\n")
    param = "Best params_/ cluster: " + str(best_cluster) + "/ eps:" + str(best_eps) + "/ min_sample:" + str(
        best_min_sample) + "/ metric:" + best_metric + "/ algorithm:" + best_algorithm + "/ leaf_size:" + str(
        best_leaf_size)
    return best_silhouette, param, list(X.columns)


"""
:param X: dataset
:param bandwidths: bandwidth used in the RBF kernel 
:param max_iters: Maximum numer of iteration
:param n_job: The number of jobs to use for the computation.
"""


def MeanShifts(X, y, bandwidths=None, max_iters=None, n_job=None):


    if bandwidths is None:
        bandwidths = [estimate_bandwidth(X, quantile=0.25), estimate_bandwidth(X, quantile=0.50), estimate_bandwidth(X, quantile=0.75)]
    if max_iters is None:
        max_iters = [200, 300, 400]
    if n_job is None:
        n_job = -1

    best_silhouette = -1
    best_cluster = -1
    best_max_iter = 0
    best_bandwidth = 0

    df = pd.DataFrame()
    centerDF = pd.DataFrame()

    for bandwidth in bandwidths:
        for max_iter in max_iters:

            clusterer = MeanShift(bandwidth=bandwidth, max_iter=max_iter, n_jobs=n_job)
            clusterer.fit(X)
            cluster_labels = clusterer.labels_
            n_clusters = len(np.unique(clusterer.labels_))

            # print(np.unique(clusterer.labels_))

            if n_clusters < 2:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            silhouette_avg = silhouette_score(X, cluster_labels)
            print("number of clusters: ", n_clusters, "/ bandwidth:", bandwidth, "/ max_iter:", max_iter)
            print("The average silhouette_score is :", silhouette_avg)

            centers = clusterer.cluster_centers_

            if best_silhouette < silhouette_avg:
                best_silhouette = silhouette_avg
                best_cluster = n_clusters
                best_bandwidth = bandwidth
                best_max_iter = max_iter

                sum = [0 for row in range(n_clusters)]
                num = [0 for row in range(n_clusters)]

                j = 0
                for i in cluster_labels:
                    sum[i] = sum[i] + y[j]
                    num[i] = num[i] + 1
                    j = j + 1

                for i in range(n_clusters):
                    sum[i] = sum[i] / num[i]
                centerDF = pd.DataFrame(centers)
                # centerDF.loc[:, 'Mean House Value'] = sum

            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                                  edgecolor=color, alpha=0.7)

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                y_lower = y_upper + 10  # 10 for the 0 samples
            ax1.set_title("Silouette Plot")
            ax1.set_xlabel("Silhouette coefficient")
            ax1.set_ylabel("Cluster label")

            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("Cluster")
            ax2.set_xlabel("1st Column")
            ax2.set_ylabel("2nd Column")

            plt.suptitle(("Silhouette analysis for MeanShift clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

            df = centerDF.copy()

    plt.show()
    print("\nThe highest silhouette score is ", best_silhouette, " with ", best_cluster, " clusters")
    print("Best params_/ bandwidth:", best_bandwidth, "/ max_iter:", best_max_iter, "\n")
    param = "Best params_/ bandwidth:" + str(best_bandwidth) + "/ max_iter:" + str(best_max_iter)
    return best_silhouette, param, list(X.columns)


"""
Do clustering and get silhouette index for various combinations of parameters
do scale and encode dataset with given scalers and encoders (select features to use for this program)
do kmeans, GMM, CLARANS, DBSCAN, MeanShift clustering to all combination datasets of encoders and scalers
do clustering with all combinations of parameters(including 'k' number of clusters)
show every result (silhouette index) and show parameters with the most highest silhouette index
"""


def findBest(or_data, y, numerical_columns, categorical_columns, max_cluster=None, n_inits=None, max_iters=None,
             tols=None, verboses=None, covariance_types=None,
             numlocals=None, max_neighbors=None, epsS=None, min_samples=None, metrics=None, algorithms=None,
             leaf_sizes=None, bandwidths=None, n_job=None):
    kmeans_best = [-1, 'column', 'scale', 'encode', 'params']
    GMM_best = [-1, 'column', 'scale', 'encode', 'params']
    DBSCAN_best = [-1, 'column', 'scale', 'encode', 'params']
    MeanShift_best = [-1, 'column', 'scale', 'encode', 'params']
    silhouette_score = 0
    params = ""

    cleaned_data = clean_outlier(or_data, ['Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration'])


    for numerical_column, categorical_column in zip(numerical_columns, categorical_columns):

        print("columns: ", numerical_column)

        total_columns = numerical_column
        x = pd.DataFrame()
        cleaned_x = pd.DataFrame()
        data = or_data.copy()


        for numerical_column_ind in numerical_column:
            x.loc[:, numerical_column_ind] = data.loc[:, numerical_column_ind]
            cleaned_x.loc[:, numerical_column_ind] = cleaned_data.loc[:, numerical_column_ind]
        for categorical_column_ind in categorical_column:
            x.loc[:, categorical_column_ind] = data.loc[:, categorical_column_ind]
            cleaned_x.loc[:, categorical_column_ind] = cleaned_data.loc[:, categorical_column_ind]


        x, scalers, encoders = get_various_encode_scale(x, numerical_column, categorical_column)
        cleaned_x, scalers, encoders = get_various_encode_scale(cleaned_x, numerical_column, categorical_column)

        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                print(scaler, encoder)
                print("--------Kmeans--------")

                silhouette_score, params, columns = kmeans(cleaned_x[i][j], y, max_cluster=max_cluster,
                                                                      n_inits=n_inits, max_iters=max_iters, tols=tols,
                                                                      verboses=verboses)
                if silhouette_score > kmeans_best[0]:
                    kmeans_best[0] = silhouette_score
                    kmeans_best[1] = columns
                    kmeans_best[2] = scaler
                    kmeans_best[3] = encoder
                    kmeans_best[4] = params

                print("--------GMM--------")

                silhouette_score, params, columns = GMM(cleaned_x[i][j], y, max_cluster=max_cluster,
                                                                covariance_types=covariance_types, tols=tols,
                                                                max_iters=max_iters, n_inits=n_inits)
                if silhouette_score > GMM_best[0]:
                    GMM_best[0] = silhouette_score
                    GMM_best[1] = columns
                    GMM_best[2] = scaler
                    GMM_best[3] = encoder
                    GMM_best[4] = params


                print("--------DBSCAN--------")

                silhouette_score, params, columns = DBSCANs(x[i][j], y, epsS=epsS, min_samples=min_samples,
                                                                       metrics=metrics, algorithms=algorithms,
                                                                       leaf_sizes=leaf_sizes)
                if silhouette_score > DBSCAN_best[0]:
                    DBSCAN_best[0] = silhouette_score
                    DBSCAN_best[1] = columns
                    DBSCAN_best[2] = scaler
                    DBSCAN_best[3] = encoder
                    DBSCAN_best[4] = params

                print("--------MeanShift--------")

                silhouette_score, params, columns = MeanShifts(x[i][j], y, bandwidths=bandwidths,
                                                                             max_iters=max_iters, n_job=n_job)
                if silhouette_score > MeanShift_best[0]:
                    MeanShift_best[0] = silhouette_score
                    MeanShift_best[1] = columns
                    MeanShift_best[2] = scaler
                    MeanShift_best[3] = encoder
                    MeanShift_best[4] = params

                j = j + 1
            i = i + 1

    # Print the Result
    print(kmeans_best)
    print(GMM_best)
    print(DBSCAN_best)
    print(MeanShift_best)


def outlier_iqr(data):
    q1, q3 = np.percentile(data, [0, 80])
    iqr= q3-q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)

    return np.where((data > upper_bound)|(data < lower_bound))

def clean_outlier(data, features):

    a = outlier_iqr(data[features[0]])[0]
    b = outlier_iqr(data[features[1]])[0]
    c = outlier_iqr(data[features[2]])[0]

    lead_outlier_index = np.concatenate((a,b,c), axis=None)

    lead_not_outlier_index = []

    for i in data.index:
        if i not in lead_outlier_index:
            lead_not_outlier_index.append(i)

    cleaned_data = data.loc[lead_not_outlier_index]
    cleaned_data = cleaned_data.reset_index(drop=True)

    return cleaned_data

"""
Function that starts automated machine learning
Compare various parameters(classifiers,scalers, encoders, columns, classifier parameters)
Return score compared given test target with predicted target 
"""
def get_Result(X, y,test_x,test_y, numerical_columns, categorical_columns, n_split_time=None,n_jobs=None):

    model, params, indices, numerical, categorical = do_Classify(X, y, numerical_columns, categorical_columns, n_split_time,n_jobs)
    print("\n=====================================")
    print("\nHighest Score:")
    print("Model: ",model)
    for param,index in zip(params,indices):
        print(index," : ",param)
    print("Numerical columns: ",numerical)
    print("Categorical columns: ",categorical)

    new_x = pd.DataFrame()
    data = X.copy()
    new_test = pd.DataFrame()
    test_data = test_x.copy()
    Foldnum = 5

    for numerical_column_ind in numerical:
        new_x.loc[:, numerical_column_ind] = data.loc[:, numerical_column_ind]
    for categorical_column_ind in categorical:
        new_x.loc[:, categorical_column_ind] = data.loc[:, categorical_column_ind]

    for numerical_column_ind in numerical:
        new_test.loc[:, numerical_column_ind] = test_data.loc[:, numerical_column_ind]
    for categorical_column_ind in categorical:
        new_test.loc[:, categorical_column_ind] = test_data.loc[:, categorical_column_ind]

    encoder = [params.pop()]
    scaler = [params.pop()]


    new_x, scalers, encoders = get_various_encode_scale(new_x, numerical, categorical,scalers=scaler,encoders=encoder)
    new_test, scalers, encoders = get_various_encode_scale(new_test, numerical, categorical, scalers=scaler, encoders=encoder)

    model_param = {}
    for param,index in zip(params,indices):
        model_param[index] = param
    model.set_params(**model_param)
    model.fit(new_x[0][0],y)

    for col in new_x[0][0].columns.difference(new_test[0][0].columns):
        new_test[0][0].loc[:,col] = 0

    y_pred = model.predict(new_test[0][0])

    compare_y = pd.DataFrame({'y_pred': y_pred, 'y_test': test_y})

    print("\nCompare Prediction and Test")
    print(compare_y)
    return ((y_pred == test_y).sum() / len(compare_y))

"""
Function that do classify with given several parameters
Return parameters that got most highest score.
"""
def do_Classify(X, y, numerical_columns, categorical_columns, n_split_time=None,n_jobs=None):

    if n_jobs == None:
        n_jobs = -1
    if n_split_time == None:
        n_split_time = 5

    DT_best_score = 0
    DT_best_param = [0, 0, 'scaler', 'encoder']
    DT_best_index = ['max_depth', 'min_samples_split', 'scaler', 'encoder']
    DT_best_numerical_cols = []
    DT_best_categorical_cols = []

    LR_best_score = 0
    LR_best_param = [0, 0, 0, 'scaler', 'encoder']
    LR_best_index = ['max_iter', 'C','tol', 'scaler', 'encoder']
    LR_best_numerical_cols = []
    LR_best_categorical_cols = []

    RF_best_score = 0
    RF_best_param = [0, 0, 0, 'scaler', 'encoder']
    RF_best_index = ['n_estimators', 'max_features', 'max_depth', 'scaler', 'encoder']
    RF_best_numerical_cols = []
    RF_best_categorical_cols = []

    for numerical_column, categorical_column in zip(numerical_columns, categorical_columns):

        new_x = pd.DataFrame()
        data = X.copy()
        Foldnum = 5

        for numerical_column_ind in numerical_column:
            new_x.loc[:, numerical_column_ind] = data.loc[:, numerical_column_ind]
        for categorical_column_ind in categorical_column:
            new_x.loc[:, categorical_column_ind] = data.loc[:, categorical_column_ind]

        new_x, scalers, encoders = get_various_encode_scale(new_x, numerical_column, categorical_column)

        i=0
        for scaler in scalers:
           j=0
           for encoder in encoders:

              x_train, x_test, y_train, y_test = train_test_split(new_x[i][j], y, test_size=1 / Foldnum, shuffle=True)

              max_depth,min_sample_split,get_f1_score = get_Best_Decision_Tree_Classifier(x_train,x_test,y_train,y_test)
              if(get_f1_score>DT_best_score):
                  DT_best_param[0] = max_depth
                  DT_best_param[1] = min_sample_split
                  DT_best_param[2] = scaler
                  DT_best_param[3] = encoder
                  DT_best_score = get_f1_score
                  DT_best_numerical_cols = numerical_column
                  DT_best_categorical_cols = categorical_column

              max_iter, C, tol,get_f1_score = get_Best_Logistic_Regression(x_train, x_test, y_train, y_test)
              if (get_f1_score > LR_best_score):
                  LR_best_param[0] = max_iter
                  LR_best_param[1] = C
                  LR_best_param[2] = tol
                  LR_best_param[3] = scaler
                  LR_best_param[4] = encoder
                  LR_best_score = get_f1_score
                  LR_best_numerical_cols = numerical_column
                  LR_best_categorical_cols = categorical_column

              n_estimator, max_features, max_depth, get_f1_score = get_Best_Random_Forest(x_train, x_test, y_train, y_test)
              if (get_f1_score > RF_best_score):
                  RF_best_param[0] = n_estimator
                  RF_best_param[1] = max_features
                  RF_best_param[2] = max_depth
                  RF_best_param[3] = scaler
                  RF_best_param[4] = encoder
                  RF_best_score = get_f1_score
                  RF_best_numerical_cols = numerical_column
                  RF_best_categorical_cols = categorical_column

              j=j+1
           i=i+1

    print("Best Score and parameters for each classifiers\n=========================================")
    print("Decision Tree")
    print("Score: ",DT_best_score)
    for index,param in zip(DT_best_index,DT_best_param):
        print(index," : ", param)
    print("Columns: ",DT_best_numerical_cols,DT_best_categorical_cols)

    print("\nLogistic Regression")
    print("Score: ", LR_best_score)
    for index, param in zip(LR_best_index, LR_best_param):
        print(index, " : ", param)
    print("Columns: ", LR_best_numerical_cols, LR_best_categorical_cols)

    print("\nRandom Forest")
    print("Score: ", RF_best_score)
    for index, param in zip(RF_best_index, RF_best_param):
        print(index, " : ", param)
    print("Columns: ", RF_best_numerical_cols, RF_best_categorical_cols)

    if(DT_best_score>=LR_best_score and DT_best_score>=RF_best_score):
        return DecisionTreeClassifier(), DT_best_param,DT_best_index ,DT_best_numerical_cols,DT_best_categorical_cols
    elif(LR_best_score>=DT_best_score and LR_best_score>=RF_best_score):
        return LogisticRegression(), LR_best_param, LR_best_index, LR_best_numerical_cols, LR_best_categorical_cols
    elif(RF_best_score>DT_best_score and RF_best_score>=LR_best_score):
        return RandomForestClassifier(),RF_best_param, RF_best_index, RF_best_numerical_cols, RF_best_categorical_cols

"""
Function that find classifier parameter of decision tree classifier that gives highest score
Do Parameter tuning with default parameters and parameter growers
If the parameter does not change by more than the minimum amount change, stop tuning
"""
def get_Best_Decision_Tree_Classifier(X_train,X_test,y_train,y_test):

    max_depth = 10
    max_depth_grower = 2
    min_samples_split = 10
    min_samples_split_grower = 2
    pre_score = 0.0
    max_score= 0.0
    direction = 1
    #minimum amount change
    MAC = 0.000001

    max_score = get_DT_score(X_train,X_test,y_train,y_test,max_depth,min_samples_split)
    left_score = get_DT_score(X_train,X_test,y_train,y_test,max_depth-max_depth_grower,min_samples_split)
    right_score = get_DT_score(X_train,X_test,y_train,y_test,max_depth+max_depth_grower,min_samples_split)

    if(max_score>left_score and max_score>right_score):
        direction = 0
    elif(left_score>right_score):
        direction = -1
    elif(right_score>left_score):
        direction = 1

    while(max_score>pre_score and ((max_score-pre_score)/max_score)>MAC and max_depth + direction * max_depth_grower>0):
        pre_score = max_score
        max_depth = max_depth + direction * max_depth_grower
        max_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth, min_samples_split)

    max_score = pre_score
    max_depth = max_depth - direction * max_depth_grower

    left_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth,
                              min_samples_split-min_samples_split_grower)
    right_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth,
                               min_samples_split+min_samples_split_grower)

    if (max_score > left_score and max_score > right_score):
        direction = 0
    elif (left_score > right_score):
        direction = -1
    elif (right_score > left_score):
        direction = 1

    pre_score = 0

    while (max_score > pre_score and ((max_score-pre_score)/max_score)>MAC and min_samples_split + direction * min_samples_split_grower>0):
        if(min_samples_split + direction * min_samples_split_grower ==0):
            min_samples_split = min_samples_split + direction * min_samples_split_grower
            break;
        pre_score = max_score
        min_samples_split = min_samples_split + direction * min_samples_split_grower
        max_score = get_DT_score(X_train, X_test, y_train, y_test, max_depth, min_samples_split)

    max_score = pre_score
    min_samples_split = min_samples_split - direction * min_samples_split_grower

    return max_depth,min_samples_split,max_score

def get_DT_score(X_train,X_test,y_train,y_test,max_depth,min_samples_split):

    model = DecisionTreeClassifier(random_state=47, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

"""
Function that find classifier parameter of Logistic regression that gives highest score
Do Parameter tuning with default parameters and parameter growers
If the parameter does not change by more than the minimum amount change, stop tuning
"""
def get_Best_Logistic_Regression(X_train,X_test,y_train,y_test):

    penalty = 'l2'
    solver = 'liblinear'
    max_iter = 300
    max_iter_grower = 5
    C = 10
    C_grower = 2
    tol = 1e-4
    tol_grower = 0.5
    direction = 1
    #minimum amount change
    MAC = 0.00001
    max_score = 0.0
    pre_score = 0.0

    #Max_iter
    max_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter,C,tol)
    left_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter-max_iter_grower,C,tol)
    right_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter+max_iter_grower,C,tol)

    if(max_score>=left_score and max_score>=right_score):
        direction = 0
    elif(left_score>right_score):
        direction = -1
    elif(right_score>left_score):
        direction = 1

    while(max_score>pre_score and ((max_score-pre_score)/max_score)>MAC and max_iter + direction * max_iter_grower > 0):
        pre_score = max_score
        max_iter = max_iter + direction * max_iter_grower
        max_score = get_LR_score(X_train, X_test, y_train, y_test,penalty,solver, max_iter, C,tol)
    max_score = pre_score
    max_iter = max_iter - direction * max_iter_grower

    # C
    left_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C-C_grower,tol)
    right_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C+C_grower,tol)

    if (max_score >= left_score and max_score >= right_score):
        direction = 0
    elif (left_score > right_score):
        direction = -1
    elif (right_score > left_score):
        direction = 1

    pre_score = 0
    while (max_score > pre_score and ((max_score-pre_score)/max_score)>MAC and C + direction * C_grower>0):
        pre_score = max_score
        C = C + direction * C_grower
        max_score = get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter,C,tol)

    max_score = pre_score
    C = C - direction * C_grower

    # tol
    left_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C,tol*tol_grower)
    right_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C,tol/tol_grower)

    if (max_score >= left_score and max_score >= right_score):
        direction = 0
    elif (right_score > left_score):
        tol_grower = 1/tol_grower

    while (max_score > pre_score and ((max_score - pre_score) / max_score) > MAC and tol * tol_grower>0):
        pre_score = max_score
        tol = tol * tol_grower
        max_score = get_LR_score(X_train, X_test, y_train, y_test, penalty, solver, max_iter, C,tol)
    max_score = pre_score
    tol = tol / max_iter_grower

    return max_iter,C,tol,max_score

def get_LR_score(X_train,X_test,y_train,y_test,penalty,solver,max_iter,C,tol):
    model = LogisticRegression(penalty=penalty,solver=solver, max_iter=max_iter,C=C,tol=tol)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')



"""
Function that find classifier parameter of random forest classifier that gives highest score
Do Parameter tuning with default parameters and parameter growers
If the parameter does not change by more than the minimum amount change, stop tuning
"""
def get_Best_Random_Forest(X_train,X_test,y_train,y_test):

    criterion = 'gini'
    n_estimators = 50
    n_estimators_grower = 5
    max_features = int(len(X_train.columns)/2)
    max_features_grower = 1
    max_depth = 50
    max_depth_grower=5

    direction = 1
    #minimum amount change
    MAC = 0.0000001
    max_score = 0.0
    pre_score = 0.0

    #n_estimator
    max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)
    left_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators-n_estimators_grower,max_features,max_depth)
    right_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators+n_estimators_grower,max_features,max_depth)

    if(max_score>=left_score and max_score>=right_score):
        direction = 0
    elif(left_score>right_score):
        direction = -1
    elif(right_score>left_score):
        direction = 1

    while(max_score>pre_score and ((max_score-pre_score)/max_score)>MAC and  n_estimators + direction * n_estimators_grower>0):
        pre_score = max_score
        n_estimators = n_estimators + direction * n_estimators_grower
        max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)
    max_score = pre_score
    n_estimators = n_estimators - direction * n_estimators_grower

    # max_features
    if (max_features + direction * max_features_grower > len(X_train.columns) and max_features + direction * max_features_grower == 0):
        left_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features-max_features_grower,max_depth)
        right_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features+max_features_grower,max_depth)

        if (max_score >= left_score and max_score >= right_score):
            direction = 0
        elif (left_score > right_score):
            direction = -1
        elif (right_score > left_score):
            direction = 1

        pre_score = 0
        while (max_score > pre_score and ((max_score-pre_score)/max_score)>MAC):
            if (max_features + direction * max_features_grower>len(X_train.columns) or max_features + direction * max_features_grower==0):
                max_features = max_features + direction * max_features_grower
                break
            pre_score = max_score
            max_features = max_features + direction * max_features_grower
            max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)

        max_score = pre_score
        max_features = max_features - direction * max_features


    # max_depth
    left_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth-max_depth_grower)
    right_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth+max_depth_grower)

    if (max_score >= left_score and max_score >= right_score):
        direction = 0
    elif (left_score > right_score):
        direction = -1
    elif (right_score > left_score):
        direction = 1

    while (max_score > pre_score and ((max_score - pre_score) / max_score) > MAC and max_depth * max_depth_grower>0):
        pre_score = max_score
        max_depth = max_depth * max_depth_grower
        max_score = get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth)
    max_score = pre_score
    max_depth = max_depth - direction * max_depth

    return n_estimators,max_features,max_depth,max_score

def get_RF_score(X_train,X_test,y_train,y_test,n_estimators,max_features,max_depth):
    model = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')