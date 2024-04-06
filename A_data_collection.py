from sklearn.datasets import make_regression, make_classification, make_blobs


# DEFAULT PARAMETERS FOR CLASSIFICATION
#X, y = make_dataset_for_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, seed=2023)

def make_dataset_for_classification(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, n_clusters_per_class, weights, flip_y, class_sep, hypercube, seed):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
                               weights=weights, flip_y=flip_y, class_sep=class_sep, hypercube=hypercube, random_state=seed)
    return X, y





# DEFAULT PARAMETERS FOR REGRESSION
#X, y = make_dataset_for_regression(n_samples=1000, n_features=3, n_informative=3, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, seed=2023)

def make_dataset_for_regression(n_samples, n_features, n_informative, n_targets, bias, effective_rank, tail_strength, noise, seed):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_targets,
                           bias=bias, effective_rank=effective_rank, tail_strength=tail_strength, noise=noise, random_state=seed)
    return X, y





# DEFAULT PARAMETERS FOR CLASSIFICATION
#X = make_dataset_for_clustering(n_samples=1000, n_features=4, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), seed=2023)

def make_dataset_for_clustering(n_samples, n_features, centers, cluster_std, center_box, seed):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, center_box=center_box, random_state=seed)
    return X
