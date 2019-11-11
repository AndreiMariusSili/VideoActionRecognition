import typing as tp

import options.data_options as do
import options.experiment_options as eo
import options.job_options as jo
import options.model_options as mo
import specs

OPTIONS = tp.Union[jo.RunExperimentOptions, jo.EvaluateExperimentOptions, jo.VisualiseOptions]


def _build_spec(opts: OPTIONS) -> eo.ExperimentOptions:
    """Build the spec from the spec name provided."""
    name = opts.spec.strip()

    dataset, cut, model, model_type = opts.spec.strip().split('/')
    model_options: mo.MODEL_OPTS = getattr(specs.models, f'{model}_{model_type}')
    databunch_options: do.DataBunchOptions = getattr(specs.datasets, dataset).dbo

    if cut == 'full':
        databunch_options.cut = 1.00
    elif cut == '3q':
        databunch_options.cut = 0.75
    elif cut == 'half':
        databunch_options.cut = 0.50
    else:
        raise ValueError(f'Unknown cut: {cut}.')

    if model_options.type == 'class':
        trainer_opts = specs.trainers.class_trainer
        evaluator_opts = specs.evaluators.class_evaluator
    elif model_options.type == 'ae':
        trainer_opts = specs.trainers.class_ae_trainer
        evaluator_opts = specs.evaluators.class_ae_evaluator
    elif model_options.type == 'vae':
        trainer_opts = specs.trainers.class_vae_trainer
        evaluator_opts = specs.evaluators.class_vae_evaluator
    else:
        raise ValueError(f'Unknown model type: {model_options.type}.')

    if opts.overfit:
        trainer_opts.epochs = 16
        databunch_options.dev_dso.do.meta_path = databunch_options.train_dso.do.meta_path
        databunch_options.test_dso.do.meta_path = databunch_options.train_dso.do.meta_path

        databunch_options.train_dso.do.keep = 128
        databunch_options.dev_dso.do.keep = 128
        databunch_options.test_dso.do.keep = 128

    if opts.dev:
        opts.debug = True
        trainer_opts.epochs = 2

        databunch_options.dlo.batch_size = 2
        databunch_options.dlo.num_workers = 1

        databunch_options.train_dso.do.keep = 16
        databunch_options.dev_dso.do.keep = 16
        databunch_options.test_dso.do.keep = 16

        databunch_options.train_dso.do.keep = 8
        databunch_options.dev_dso.do.keep = 8
        databunch_options.test_dso.do.keep = 8

    exp_opts = eo.ExperimentOptions(
        name=name,
        resume=opts.resume,
        debug=opts.debug,
        model_opts=model_options,
        databunch_opts=databunch_options,
        trainer=trainer_opts,
        evaluator=evaluator_opts
    )

    return exp_opts


def setup(opts: jo.SetupOptions) -> None:
    """Run dataset setup."""
    if opts.set == 'smth':
        import prepro.smth
        prepro.smth.setup()
    elif opts.set == 'hmdb':
        import prepro.hmdb
        prepro.hmdb.setup()
    else:
        raise ValueError(f'Unknown options: {opts}')


def create_dummy_set(opts: jo.CreateDummySetOptions) -> None:
    """Create a dummy subset of the full dataset specified in opts."""
    if opts.set == 'smth':
        import prepro.smth
        prepro.smth.create_dummy_set(int(opts.split))
    else:
        raise ValueError(f'Unknown options: {opts}')


def prepro_set(opts: jo.PreproSetOptions) -> None:
    """Preprocess the dataset specified in opts in predefined ways."""
    import prepro
    prepro.common.gather_dim_stats(opts.set, int(opts.split))
    prepro.common.split_train_dev(opts.set, int(opts.split))
    prepro.common.augment_meta(opts.set, int(opts.split))
    prepro.common.merge_meta(opts.set, int(opts.split))
    if opts.jpeg:
        prepro.common.extract_jpeg(opts.set, int(opts.split))


def run_experiment(opts: jo.RunExperimentOptions):
    """Run experiment."""
    spec: eo.ExperimentOptions = _build_spec(opts)
    if spec.model_opts.type == 'class':
        import pro.runners.runner_class as rcl
        rcl.ClassRunner(spec, opts.local_rank).run()
    elif spec.model_opts.type == 'ae':
        import pro.runners.runner_ae as rae
        rae.AutoEncoderRunner(spec, opts.local_rank).run()
    elif spec.model_opts.type == 'vae':
        import pro.runners.runner_vae as rvae
        rvae.VariationalAutoEncoderRunner(spec, opts.local_rank).run()


def evaluate_model(opts: jo.EvaluateExperimentOptions):
    """Gather results for a model and store in a data frame."""
    import pro.evaluators._evaluator_base as pe
    spec = _build_spec(opts)

    pe.BaseEvaluator(spec, opts.local_rank).start()


def visualise_model(opts: jo.VisualiseOptions):
    """Create bokeh visualisation for a trained model."""
    import postpro.visualisation as pv
    page = opts.page
    spec = _build_spec(opts)
    pv.Visualisation(page, spec).start()
