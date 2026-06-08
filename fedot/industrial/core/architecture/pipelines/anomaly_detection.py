from functools import partial
import pandas as pd
from pymonad.list import ListMonad
from pymonad.either import Right
from fedot.industrial.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot.industrial.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation


class AnomalyDetectionPipelines(AbstractPipeline):

    def __call__(self, pipeline_type: str = 'SpecifiedFeatureGeneratorTSC'):
        pipeline_dict = {'SST': self.__singular_transformation_pipeline,
                         'FunctionalPCA': self.__functional_pca_pipeline,
                         'Kalman': self.__kalman_filter_pipeline,
                         'VectorAngle': self.__vector_based_pipeline,
                         }
        return pipeline_dict[pipeline_type]

    def get_feature_generator(self, **kwargs):
        lambda_func_dict = kwargs['steps']
        lambda_func_dict['concatenate_features'] = lambda x: list_of_features.append(
            pd.DataFrame(x))
        input_data = kwargs['input_data']
        list_of_features = []
        if kwargs['pipeline_type'] == 'SST':
            pipeline = Right(input_data).then(lambda_func_dict['fit_model'])
        elif kwargs['pipeline_type'] == 'FunctionalPCA':
            if not isinstance(input_data, pd.DataFrame):
                input_data = pd.DataFrame(input_data)
            pipeline = Right(input_data).then(
                lambda_func_dict['create_list_of_ts']).map(
                lambda_func_dict['transform_to_basis']).then(
                lambda_func_dict['reduce_basis']).then(
                lambda_func_dict['concatenate_features']).insert(
                    self._get_feature_matrix(
                        list_of_features=list_of_features,
                        mode='1D'))
        elif kwargs['pipeline_type'] == 'Kalman':
            pipeline = Right(input_data).then(
                lambda_func_dict['fit_model']).insert(
                self._get_feature_matrix(
                    list_of_features=self.test_features,
                    mode='list_of_ts',
                    window_length=kwargs['window_length']))
        elif kwargs['pipeline_type'] == 'VectorAngle':
            pipeline = Right(input_data)
        return pipeline

    def __singular_transformation_pipeline(
            self, mode: str = 'multits', **kwargs):
        feature_extractor, detector, lambda_func_dict = self._init_pipeline_nodes(
            model_type='sst', **kwargs)

        if mode == 'multits':
            self.test_features = [pd.DataFrame(
                time_series) for time_series in self.test_features]
            self.test_features = [time_series.values.tolist()
                                  for time_series in self.test_features]

        lambda_func_dict['fit_model'] = lambda list_of_ts: [
            detector.fit(train_features=slice_ts) for slice_ts in list_of_ts]
        lambda_func_dict['predict'] = lambda list_of_ts: [detector.predict(
            test_features=slice_ts[0], target=slice_ts[1]) for slice_ts in list_of_ts]

        train_feature_generator_module = self.get_feature_generator(
            input_data=self.train_features, steps=lambda_func_dict, pipeline_type='SST')
        prediction = train_feature_generator_module.map(
            lambda_func_dict['predict'])

        detector.singular_transformation = partial(self.get_feature_generator,
                                                   steps=lambda_func_dict,
                                                   pipeline_type='SST')
        return detector, prediction.value

    def __multi_functional_pca(self):
        pass

    def __functional_pca_pipeline(self, decomposition: str = 'svd', **kwargs):
        feature_extractor, detector, lambda_func_dict = self._init_pipeline_nodes(
            model_type='functional_pca', **kwargs)
        data_basis = EigenBasisImplementation()
        if kwargs['component'] is None:
            kwargs['component'] = [0 for i in range(
                self.train_features.shape[1])]
        if decomposition == 'tensor':
            lambda_func_dict['create_list_of_ts'] = lambda time_series: ListMonad(
                time_series.values.tolist())
        else:
            lambda_func_dict['create_list_of_ts'] = lambda time_series: ListMonad(
                *time_series.values.tolist())

        lambda_func_dict['transform_to_basis'] = lambda time_series: ListMonad(
            self.basis if self.basis is not None else data_basis.fit(
                time_series, window_length=None))
        lambda_func_dict['reduce_basis'] = lambda list_of_components: [
            component[:, rank] for component, rank in zip(list_of_components, kwargs['component'])]
        lambda_func_dict['predict'] = lambda list_of_ts: [
            detector.predict(test_features=slice_ts) for slice_ts in list_of_ts]

        train_feature_generator_module = self.get_feature_generator(
            input_data=self.train_features,
            steps=lambda_func_dict,
            pipeline_type='FunctionalPCA')
        detector.fit(train_feature_generator_module.value[0])

        prediction = train_feature_generator_module.insert(
            self.test_features).map(lambda_func_dict['predict'])
        detector.feature_generator = partial(
            self.get_feature_generator,
            steps=lambda_func_dict,
            pipeline_type='FunctionalPCA')

        return detector, prediction.value[0]

    def __kalman_filter_pipeline(self, **kwargs):

        feature_extractor, detector, lambda_func_dict = self._init_pipeline_nodes(
            model_type='kalman_filter', **kwargs)
        lambda_func_dict['fit_model'] = lambda train_features: detector.fit(
            train_features=train_features)
        lambda_func_dict['predict'] = lambda list_of_ts: [
            detector.predict(test_features=slice_ts) for slice_ts in list_of_ts]

        train_feature_generator_module = self.get_feature_generator(
            input_data=self.train_features,
            steps=lambda_func_dict,
            pipeline_type='Kalman',
            window_length=kwargs['window_length'])
        prediction = train_feature_generator_module.map(
            lambda_func_dict['predict'])

        detector.state_transition_matrix = partial(
            self.get_feature_generator,
            steps=lambda_func_dict,
            pipeline_type='Kalman',
            window_length=kwargs['window_length'])
        return detector, prediction.value

    def __vector_based_pipeline(self):
        pass
