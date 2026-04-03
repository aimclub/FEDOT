import sys
import types

try:
    import pytest
except ModuleNotFoundError:
    class _PytestStub:
        class mark:
            @staticmethod
            def skipif(condition, reason=''):
                def decorator(func):
                    return func
                return decorator

    pytest = _PytestStub()

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _install_fedcore_data_stub():
    if 'fedcore.api.utils.data' in sys.modules:
        return

    fedcore_module = types.ModuleType('fedcore')
    api_module = types.ModuleType('fedcore.api')
    utils_module = types.ModuleType('fedcore.api.utils')
    data_module = types.ModuleType('fedcore.api.utils.data')

    class _DataLoaderHandler:
        @staticmethod
        def check_convert(dataloader=None, **kwargs):
            return dataloader

    data_module.DataLoaderHandler = _DataLoaderHandler

    sys.modules['fedcore'] = fedcore_module
    sys.modules['fedcore.api'] = api_module
    sys.modules['fedcore.api.utils'] = utils_module
    sys.modules['fedcore.api.utils.data'] = data_module


_install_fedcore_data_stub()

if torch is not None:
    from fedot.industrial.core.models.nn.utils.hooks import BaseHook, EarlyStopping, Freezer
    from fedot.industrial.core.models.nn.utils.hooks_collection import HooksCollection

    @pytest.mark.skipif(torch is None, reason='torch is required for industrial hook tests')
    class TestHooksCollection:
        class _StartHook(BaseHook):
            _SUMMON_KEY = 'start_hook'
            _hook_place = -20

            def trigger(self, epoch, kws):
                return False

            def action(self, epoch, kws):
                return None

        class _EarlierStartHook(BaseHook):
            _SUMMON_KEY = 'earlier_start_hook'
            _hook_place = -50

            def trigger(self, epoch, kws):
                return False

            def action(self, epoch, kws):
                return None

        class _EndHook(BaseHook):
            _SUMMON_KEY = 'end_hook'
            _hook_place = 10

            def trigger(self, epoch, kws):
                return False

            def action(self, epoch, kws):
                return None

        class _LaterEndHook(BaseHook):
            _SUMMON_KEY = 'later_end_hook'
            _hook_place = 30

            def trigger(self, epoch, kws):
                return False

            def action(self, epoch, kws):
                return None

        class _BothStagesHook(BaseHook):
            _SUMMON_KEY = 'both_stages_hook'
            _hook_place = 0

            def trigger(self, epoch, kws):
                return False

            def action(self, epoch, kws):
                return None

        def test_base_hook_exposes_backward_compatible_hook_place_alias(self):
            hook = self._StartHook(params={}, model=None)

            assert hook.HOOK_PLACE == -20

        def test_hooks_collection_routes_and_orders_hooks_by_priority(self):
            collection = HooksCollection()
            later_start = self._StartHook(params={}, model=None)
            earlier_start = self._EarlierStartHook(params={}, model=None)
            earlier_end = self._EndHook(params={}, model=None)
            later_end = self._LaterEndHook(params={}, model=None)
            both = self._BothStagesHook(params={}, model=None)

            collection.extend([later_start, later_end, both, earlier_start, earlier_end])

            assert collection.start == [earlier_start, later_start, both]
            assert collection.end == [both, earlier_end, later_end]

        def test_hooks_collection_instances_do_not_share_default_state(self):
            first = HooksCollection()
            second = HooksCollection()

            first.append(self._StartHook(params={}, model=None))

            assert len(first.start) == 1
            assert second.start == []
            assert second.end == []

        def test_hooks_collection_rejects_non_hook_objects(self):
            collection = HooksCollection()

            try:
                collection.append(object())
            except TypeError as exc:
                assert 'BaseHook' in str(exc)
            else:
                raise AssertionError('Expected TypeError for non-hook object')

        def test_early_stopping_uses_plateau_angle_without_linear_solve_failure(self):
            hook = EarlyStopping(
                params={'early_stop_after': 1, 'horizon': 3, 'angle_tol': 1.0},
                model=torch.nn.Linear(1, 1),
            )
            trainer_objects = {'stop': False}
            history = {'train_loss': [(1, 1.0), (2, 1.0), (3, 1.0)]}

            assert hook.trigger(epoch=3, kws={'history': history}) is True

            hook.action(epoch=3, kws={'history': history, 'trainer_objects': trainer_objects})

            assert trainer_objects['stop'] is True

        def test_freezer_uses_params_dict_access_for_last_epoch_boundary(self):
            hook = Freezer(
                params={'epochs': 3, 'refreeze_each': 1, 'frozen_prop': 0.0},
                model=torch.nn.Linear(2, 1),
            )

            hook.action(epoch=3, kws={})

            assert all(parameter.requires_grad for parameter in hook.model.parameters())
else:
    @pytest.mark.skipif(True, reason='torch is required for industrial hook tests')
    class TestHooksCollection:
        def test_torch_required(self):
            assert True