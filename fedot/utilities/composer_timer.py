import datetime
from contextlib import contextmanager


class ComposerTimer:
    def __init__(self):
        self.data_definition_fit_spend_time = None
        self.data_definition_predict_spend_time = None
        self.preprocessing_spend_time = None
        self.fitting_spend_time = None
        self.predicting_spend_time = None
        self.tuning_composing_spend_time = None
        self.tuning_post_spend_time = None
        self.train_on_full_dataset_time = None
        self.compoising_spend_time = None

        self.reset_timer()

    def reset_timer(self):
        self.data_definition_fit_spend_time = datetime.timedelta(minutes=0)
        self.data_definition_predict_spend_time = datetime.timedelta(minutes=0)
        self.preprocessing_spend_time = datetime.timedelta(minutes=0)
        self.fitting_spend_time = datetime.timedelta(minutes=0)
        self.predicting_spend_time = datetime.timedelta(minutes=0)
        self.tuning_composing_spend_time = datetime.timedelta(minutes=0)
        self.tuning_post_spend_time = datetime.timedelta(minutes=0)
        self.train_on_full_dataset_time = datetime.timedelta(minutes=0)
        self.compoising_spend_time = datetime.timedelta(minutes=0)

    @contextmanager
    def launch_data_definition(self, stage: str):
        starting_time = datetime.datetime.now()
        yield

        ending_time = datetime.datetime.now()
        if stage == 'fit':
            self.data_definition_fit_spend_time += ending_time - starting_time

        elif stage == 'predict':
            self.data_definition_predict_spend_time += ending_time - starting_time

    @contextmanager
    def launch_preprocessing(self):
        starting_time = datetime.datetime.now()
        yield
        ending_time = datetime.datetime.now()
        self.preprocessing_spend_time += ending_time - starting_time

    @contextmanager
    def launch_fitting(self):
        starting_time = datetime.datetime.now()
        yield
        ending_time = datetime.datetime.now()
        self.fitting_spend_time += ending_time - starting_time

    @contextmanager
    def launch_predicting(self):
        starting_time = datetime.datetime.now()
        yield
        ending_time = datetime.datetime.now()
        self.predicting_spend_time += ending_time - starting_time

    @contextmanager
    def launch_tuning(self, stage: str):
        starting_time = datetime.datetime.now()
        yield
        ending_time = datetime.datetime.now()

        if stage == 'composing':
            self.tuning_composing_spend_time += ending_time - starting_time

        elif stage == 'post':
            self.tuning_post_spend_time += ending_time - starting_time

    @contextmanager
    def launch_train_inference(self):
        starting_time = datetime.datetime.now()
        yield
        ending_time = datetime.datetime.now()
        self.train_on_full_dataset_time += ending_time - starting_time

    @contextmanager
    def launch_composing(self):
        starting_time = datetime.datetime.now()
        yield
        ending_time = datetime.datetime.now()
        self.compoising_spend_time += ending_time - starting_time

    @property
    def report(self) -> dict:
        """ Return dict with the next columns:
            - 'Data Definition (fit)': Time spent on data definition in fit().
            - 'Data Preprocessing': Total time spent on preprocessing data, includes fitting and predicting stages.
            - 'Fitting (summary)': Total time spent on Composing, Tuning and Training Inference.
            - 'Composing': Time spent on searching for the best pipeline.
            - 'Train Inference': Time spent on training the pipeline found during composing.
            - 'Tuning (composing)': Time spent on hyperparameters tuning in whole fitting, if with_tune is True.
            - 'Tuning (after)': Time spent on .tune() (hyperparameters tuning) after composing.
            - 'Data Definition (predict)': Time spent on data definition in predict().
            - 'Predicting': Time spent on predicting (inference).
        """

        output = {
            'Data Definition (fit)': self.data_definition_fit_spend_time,
            'Data Preprocessing': self.preprocessing_spend_time,
            'Fitting (summary)': self.fitting_spend_time,
            'Composing': self.compoising_spend_time,
            'Train Inference': self.train_on_full_dataset_time,
            'Tuning (composing)': self.tuning_composing_spend_time,
            'Tuning (after)': self.tuning_post_spend_time,
            'Data Definition (predict)': self.data_definition_predict_spend_time,
            'Predicting': self.predicting_spend_time,
        }

        return output


fedot_composer_timer = ComposerTimer()
