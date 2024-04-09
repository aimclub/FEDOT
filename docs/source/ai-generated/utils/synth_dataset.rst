Synthetic Dataset Generator Functions
=====================================

Functions for generating synthetic datasets for classification and regression problems.

Functions:
    classification_dataset(): Generates a random dataset for n-class classification problem.
    regression_dataset(): Generates a random dataset for regression problem.
    gauss_quantiles_dataset(): Generates a random dataset for n-class classification problem based on multi-dimensional Gaussian distribution quantiles.
    generate_synthetic_data(): Generates a synthetic one-dimensional array without omissions.

Args:
    samples_amount (int): Total amount of samples in the resulted dataset.
    features_amount (int): Total amount of features per sample.
    classes_amount (int): The amount of classes in the dataset.
    features_options (dict): Dictionary containing features options.
    noise_fraction (float): Fraction of noisy labels in the dataset.
    full_shuffle (bool): If True, all features and samples will be shuffled.
    weights (array): Proportions of samples assigned to each class.

Returns:
    features: Features as numpy arrays.
    target: Target as numpy arrays.
