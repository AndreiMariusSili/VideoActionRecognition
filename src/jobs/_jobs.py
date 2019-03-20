from typing import Dict, Any
import os

from jobs import options
from jobs import specs
import constants as ct


def create_dummy_set(opts: options.CreateDummySetOptions) -> None:
    """Create a dummy subset of the full dataset specified in opts."""
    if opts.set == 'smth':
        import prepro.smth
        prepro.smth.create_dummy_set(opts.sample)
    else:
        raise ValueError(f'Unknown options: {opts}')


def prepro_set(opts: options.PreproSetOptions) -> None:
    """Preprocess the dataset specified in opts in predefined ways."""
    if opts.set == 'smth':
        import prepro.smth
        prepro.smth.gather_dimension_stats()
        prepro.smth.gather_dist_stats()
        prepro.smth.extract_jpeg()
        prepro.smth.augment_meta()
        prepro.smth.merge_meta()
    else:
        raise ValueError(f'Unknown options: {opts}')


def prepare_model(opts: Dict[str, Any]):
    """Prepare a model before, e.g. setup a pre-trained model."""
    if opts['model'] == 'i3d':
        import models.i3d
        opts = options.I3DPrepareOptions(**opts)
        if int(opts.rgb):
            os.makedirs(ct.I3D_PT_RGB_CHECKPOINT.parent, exist_ok=True)
            models.i3d.prepare(ct.I3D_TF_RGB_CHECKPOINT.as_posix(), ct.I3D_PT_RGB_CHECKPOINT.as_posix(),
                               batch_size=opts.batch_size, modality='rgb')
        if int(opts.flow):
            os.makedirs(ct.I3D_PT_FLOW_CHECKPOINT.parent)
            models.i3d.prepare(ct.I3D_TF_FLOW_CHECKPOINT.as_posix(), ct.I3D_PT_FLOW_CHECKPOINT.as_posix(),
                               batch_size=opts.batch_size, modality='flow')
    else:
        raise ValueError(f'Unknown model {opts["model"]}.')


def run_model(opts: options.ModelRunOptions):
    """Run model training and evaluation."""
    import models.run as mr
    spec = getattr(specs, opts.spec)
    mr.Run(spec).run()


def evaluate_model(opts: options.ModelEvaluateOptions):
    """Gather results for a model and store in a data frame."""
    import postpro.evaluation as pe
    spec = getattr(specs, opts.spec)
    pe.Evaluation(spec).start()


def visualise_model(opts: options.ModelVisualiseOptions):
    """Create bokeh visualisation for a trained model."""
    import postpro.visualisation as pv

    page = opts.page
    run_opts = getattr(specs, opts.spec)
    pv.Visualisation(page, run_opts).start()
