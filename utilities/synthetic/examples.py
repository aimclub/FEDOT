import matplotlib.pyplot as plt

from utilities.synthetic.data import classification_dataset, gauss_quantiles_dataset


def data_generator_example():
    samples_total, features_amount = 100, 10
    classes = 2
    options = {
        'informative': 8,
        'redundant': 1,
        'repeated': 1,
        'clusters_per_class': 1
    }
    features, target = classification_dataset(samples_total, features_amount, classes,
                                              features_options=options,
                                              noise_fraction=0.1, full_shuffle=False)

    plt.subplot(121)
    plt.title("Two informative features, one cluster per class")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    features, target = gauss_quantiles_dataset(samples_total, features_amount=2, classes_amount=classes)
    plt.subplot(122)
    plt.title("Gaussian divided into three quantiles")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    plt.show()


if __name__ == '__main__':
    data_generator_example()
