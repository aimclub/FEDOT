__all__ = [
    'MultiModalData',
    'SupplementaryData',
]


def __getattr__(name):
    if name == 'MultiModalData':
        from fedot.core.data.multimodal.multi_modal import MultiModalData

        return MultiModalData
    if name == 'SupplementaryData':
        from fedot.core.data.multimodal.supplementary_data import SupplementaryData

        return SupplementaryData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
