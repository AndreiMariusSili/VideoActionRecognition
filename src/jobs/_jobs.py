from jobs import _options
from prepro import smth


def create_dummy_set(opts: _options.CreateDummySetOptions) -> None:
    """Create a dummy subset of the full dataset specified in opts."""
    if opts.set == 'smth':
        smth.create_dummy_set()
    else:
        raise ValueError(f'Unknown options: {opts}')


def prepro_set(opts: _options.PreproSetOptions) -> None:
    """Preprocess the dataset specified in opts in predefined ways"""
    if opts.set == 'smth':
        smth.gather_dimension_stats()
        smth.gather_dist_stats()
        smth.augment_meta()
        smth.merge_meta()
    else:
        raise ValueError(f'Unknown options: {opts}')
