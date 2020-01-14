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
class_vae_trainer = options.experiment_options.TrainerOptions(
    epochs=128,
    lr_milestones=[32, 64, 96],
    lr_gamma=0.5,
    kld_warmup_epochs=16,
    optim_opts=options.experiment_options.AdamOptimizerOptions(),
    criterion='vae_criterion',
    metrics='train_vae_metrics'
)
