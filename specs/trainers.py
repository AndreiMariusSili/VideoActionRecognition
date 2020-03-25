import options.experiment_options

########################################################################################################################
# TRAINERS
########################################################################################################################
class_trainer = options.experiment_options.TrainerOptions(
    epochs=128,
    lr_milestones=[32, 64, 96],
    lr_gamma=0.5,
    kld_warmup_epochs=None,
    optim_opts=options.experiment_options.AdamOptimizerOptions(),
    criterion='class_criterion',
    metrics='train_class_metrics'
)
class_ae_trainer = options.experiment_options.TrainerOptions(
    epochs=128,
    lr_milestones=[32, 64, 96],
    lr_gamma=0.5,
    kld_warmup_epochs=None,
    optim_opts=options.experiment_options.AdamOptimizerOptions(),
    criterion='ae_criterion',
    metrics='train_ae_metrics'
)
class_gsnn_trainer = options.experiment_options.TrainerOptions(
    epochs=256,
    lr_milestones=[128, 162, 194],
    lr_gamma=0.5,
    kld_warmup_epochs=256,
    optim_opts=options.experiment_options.AdamOptimizerOptions(),
    criterion='gsnn_criterion',
    metrics='train_gsnn_metrics'
)
class_vae_trainer = options.experiment_options.TrainerOptions(
    epochs=256,
    lr_milestones=[128, 162, 194],
    lr_gamma=0.5,
    kld_warmup_epochs=256,
    optim_opts=options.experiment_options.AdamOptimizerOptions(),
    criterion='vae_criterion',
    metrics='train_vae_metrics'
)
