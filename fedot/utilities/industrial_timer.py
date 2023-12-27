import datetime
from contextlib import contextmanager


class FedotIndustrialTimer:
    def __init__(self):
        self.data_definition_fit_spend_time = datetime.timedelta(minutes=0)
        self.data_definition_predict_spend_time = datetime.timedelta(minutes=0)

        self.applying_recs_fit_spend_time = datetime.timedelta(minutes=0)
        self.applying_recs_predict_spend_time = datetime.timedelta(minutes=0)

        self.preprocessing_spend_time = datetime.timedelta(minutes=0)

        self.fitting_spend_time = datetime.timedelta(minutes=0)
        self.predicting_spend_time = datetime.timedelta(minutes=0)
        self.tuning_spend_time = datetime.timedelta(minutes=0)

        self.spend_time = {
            'Data Definition (fit)': self.data_definition_fit_spend_time,
            'Applying Recommendation (fit)': self.applying_recs_fit_spend_time,
            'Data Preprocessing': self.predicting_spend_time,
            'Fitting': self.fitting_spend_time,
            'Tuning': self.tuning_spend_time,
            'Data Definition (predict)': self.data_definition_predict_spend_time,
            'Applying Recommendation (predict)': self.applying_recs_predict_spend_time,
            'Predicting': self.predicting_spend_time,
        }

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
    def launch_tuning(self):
        starting_time = datetime.datetime.now()
        yield
        ending_time = datetime.datetime.now()
        self.tuning_spend_time += ending_time - starting_time

    @property
    def report(self):
        return self.spend_time


fedot_ind_timer = FedotIndustrialTimer()
