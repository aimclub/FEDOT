import datetime
from contextlib import contextmanager


class ComposerTimer:
    def __init__(self):
        self.data_definition_fit_spend_time = None
        self.data_definition_predict_spend_time = None
        self.applying_recs_fit_spend_time = None
        self.applying_recs_predict_spend_time = None
        self.preprocessing_spend_time = None
        self.fitting_spend_time = None
        self.predicting_spend_time = None
        self.tuning_composing_spend_time = None
        self.tuning_post_spend_time = None

        self.reset_timer()

    def reset_timer(self):
        self.data_definition_fit_spend_time = datetime.timedelta(minutes=0)
        self.data_definition_predict_spend_time = datetime.timedelta(minutes=0)
        self.applying_recs_fit_spend_time = datetime.timedelta(minutes=0)
        self.applying_recs_predict_spend_time = datetime.timedelta(minutes=0)
        self.preprocessing_spend_time = datetime.timedelta(minutes=0)
        self.fitting_spend_time = datetime.timedelta(minutes=0)
        self.predicting_spend_time = datetime.timedelta(minutes=0)
        self.tuning_composing_spend_time = datetime.timedelta(minutes=0)
        self.tuning_post_spend_time = datetime.timedelta(minutes=0)

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
    def launch_applying_recommendations(self, stage: str):
        starting_time = datetime.datetime.now()
        yield

        ending_time = datetime.datetime.now()
        if stage == 'fit':
            self.applying_recs_fit_spend_time += ending_time - starting_time

        elif stage == 'predict':
            self.applying_recs_predict_spend_time += ending_time - starting_time

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

    @property
    def report(self):
        output = {
            'Data Definition (fit)': self.data_definition_fit_spend_time,
            'Applying Recommendation (fit)': self.applying_recs_fit_spend_time,
            'Data Preprocessing': self.preprocessing_spend_time,
            'Fitting': self.fitting_spend_time,
            'Tuning (fit)': self.tuning_composing_spend_time,
            'Tuning (post)': self.tuning_post_spend_time,
            'Data Definition (predict)': self.data_definition_predict_spend_time,
            'Applying Recommendation (predict)': self.applying_recs_predict_spend_time,
            'Predicting': self.predicting_spend_time,
        }

        return output


fedot_composer_timer = ComposerTimer()
