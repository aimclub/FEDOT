from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def test_generate_data_uni():
    generator = TimeSeriesDatasetsGenerator(num_samples=80,
                                            max_ts_len=50,
                                            binary=False,
                                            test_size=0.5)
    (X_train, y_train), (X_test, y_test) = generator.generate_data()

    assert X_train.shape[0] == X_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]


def test_generate_data_multi():
    generator = TimeSeriesDatasetsGenerator(num_samples=80,
                                            max_ts_len=50,
                                            binary=False,
                                            test_size=0.5,
                                            multivariate=True)
    (X_train, y_train), (X_test, y_test) = generator.generate_data()

    assert X_train.shape[0] == X_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
