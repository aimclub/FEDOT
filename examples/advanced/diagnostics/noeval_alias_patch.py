"""Build an isolated CatBoost no-eval-set compatibility candidate."""


def patched_source(source):
    prefix, body = source.split('class FedotCatBoostImplementation(ModelImplementation):', 1)
    old = "        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}"
    new = """        # No validation set means no overfitting detector. Do not pass its
        # aliases together, and do not synthesize early_stopping_rounds=False.
        ignored_eval_params = ({'early_stopping_rounds', 'od_wait', 'od_type', 'od_pval'}
                               if not self.params.get('use_eval_set') else set())
        self.model_params = {k: v for k, v in self.params.to_dict().items()
                             if k not in self.__operation_params and k not in ignored_eval_params}"""
    assert body.count(old) == 1
    body = body.replace(old, new)
    old = """        use_best_model = self.params.get('use_best_model')
        early_stopping_rounds = self.params.get('early_stopping_rounds')
        use_eval_set = self.params.get('use_eval_set')

        if (use_best_model or isinstance(early_stopping_rounds, int)) and not use_eval_set:
            self.params.update(use_best_model=False, early_stopping_rounds=False)"""
    new = """        if not self.params.get('use_eval_set'):
            self.params.update(use_best_model=False)"""
    assert body.count(old) == 1
    return prefix + 'class FedotCatBoostImplementation(ModelImplementation):' + body.replace(old, new)
