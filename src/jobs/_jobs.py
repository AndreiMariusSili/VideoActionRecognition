from typing import Union

import options.model_options as mo
from jobs import specs
from options import job_options

OPTIONS = Union[job_options.ModelRunOptions, job_options.ModelEvaluateOptions, job_options.ModelVisualiseOptions]


def _get_spec(opts: OPTIONS) -> mo.RunOptions:
    """Get the spec from the appended group."""
    name_maybe_factors = opts.spec.split("@")
    name = name_maybe_factors.pop(0)
    group = f'__{name.split("_").pop()}'
    group = getattr(specs, group)
    spec = getattr(group, name)

    if name_maybe_factors:
        factors = name_maybe_factors.pop(0).split('_')

        if len(factors) == 2:
            mse, ce = (float(factor) for factor in factors)
            spec.trainer_opts.criterion_opts.mse_factor = mse
            spec.trainer_opts.criterion_opts.ce_factor = ce
        elif len(factors) == 3:
            mse, ce, kld = (float(factor) for factor in factors)
            spec.trainer_opts.criterion_opts.mse_factor = mse
            spec.trainer_opts.criterion_opts.ce_factor = ce
            spec.trainer_opts.criterion_opts.kld_factor = kld

    return spec


def setup(opts: job_options.SetupOptions) -> None:
    if opts.set == 'smth':
        import prepro.smth
        prepro.smth.setup()
    else:
        raise ValueError(f'Unknown options: {opts}')


def create_dummy_set(opts: job_options.CreateDummySetOptions) -> None:
    """Create a dummy subset of the full dataset specified in opts."""
    if opts.set == 'smth':
        import prepro.smth
        prepro.smth.create_dummy_set()
    else:
        raise ValueError(f'Unknown options: {opts}')


def prepro_set(opts: job_options.PreproSetOptions) -> None:
    """Preprocess the dataset specified in opts in predefined ways."""
    if opts.set == 'smth':
        import prepro.smth
        prepro.smth.gather_dimension_stats()
        prepro.smth.gather_dist_stats()
        prepro.smth.augment_meta()
        prepro.smth.split_train_dev()
        prepro.smth.merge_meta()
        if opts.jpeg:
            prepro.smth.extract_jpeg()
    else:
        raise ValueError(f'Unknown options: {opts}')


def run_model(opts: job_options.ModelRunOptions):
    """Run model training and evaluation."""
    import models.run as mr
    spec = _get_spec(opts)
    spec.resume = opts.resume
    mr.Run(spec, opts.local_rank).run()


def evaluate_model(opts: job_options.ModelEvaluateOptions):
    """Gather results for a model and store in a data frame."""
    import postpro.evaluation as pe
    spec = _get_spec(opts)

    pe.Evaluation(spec, opts.local_rank).start()


def visualise_model(opts: job_options.ModelVisualiseOptions):
    """Create bokeh visualisation for a trained model."""
    import postpro.visualisation as pv
    page = opts.page
    spec = _get_spec(opts)
    pv.Visualisation(page, spec).start()
