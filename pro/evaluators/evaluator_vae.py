import typing as typ

import ignite.engine as ie
import pandas as pd
import torch as th
import torch.utils.data as tud
import tqdm

import constants as ct
import options.data_options as do
import options.experiment_options as eo
import pro.engine as pe
import pro.evaluators._evaluator_base as _base


class ClassEvaluator(_base.BaseEvaluator):
    def __init__(self, opts: eo.ExperimentOptions, local_rank: int):
        super(ClassEvaluator, self).__init__(opts, local_rank)

    def _init_evaluators(self) -> typ.Tuple[ie.Engine, ie.Engine, ie.Engine]:
        train_evaluator = pe.create_vae_evaluator(self.model, self.opts.evaluator.metrics, self.device, True,
                                                  ct.VAE_NUM_SAMPLES_TEST)
        dev_evaluator = pe.create_vae_evaluator(self.model, self.opts.evaluator.metrics, self.device, True,
                                                ct.VAE_NUM_SAMPLES_TEST)
        test_evaluator = pe.create_vae_evaluator(self.model, self.opts.evaluator.metrics, self.device, True,
                                                 ct.VAE_NUM_SAMPLES_TEST)

        return train_evaluator, dev_evaluator, test_evaluator

    def _calculate_results(self, evaluator: ie.Engine, loader: tud.DataLoader, options: do.DataBunchOptions,
                           results: pd.DataFrame, split: str):
        """Calculate extra results: predictions, embeddings, etc."""
        loader.dataset.evaluating = True
        ids, targets, embeds, energies, means, variances, confs = [], [], [], [], [], [], []
        pbar = tqdm.tqdm(total=len(loader.dataset))

        with th.no_grad():
            for i, (video_data, video_labels, videos, labels) in enumerate(loader):
                x = video_data.to(device=self.device, non_blocking=True)

                ids.extend([video.meta.id for video in videos])
                targets.extend([label.data for label in labels])

                recon, energy, embed, mean, variance, vote = self.model(x, True, ct.VAE_NUM_SAMPLES_TEST)

                embeds.append(embed.cpu())
                energies.append(energy.cpu())
                confs.append(vote.cpu())

                pbar.update(x.shape[0])

        confs = th.cat(tuple(confs), dim=0).numpy()
        conf5, pred5 = confs.topk(5, dim=-1)
        results.loc[ids, self.CONF_COLS] = conf5.reshape(-1, 5)
        results.loc[ids, self.PRED_COLS] = pred5.reshape(-1, 5)

        embeds = th.cat(tuple(embeds), dim=0).numpy()
        energies = th.cat(tuple(energies), dim=0).numpy()

        embeds = pd.DataFrame(data=embeds, index=results.index)
        energies = pd.DataFrame(data=energies, index=results.index)

        embeds.to_parquet((self.run_dir / f'embeds_{split}.pqt').as_posix(), engine='pyarrow', index=True)
        energies.to_parquet((self.run_dir / f'energies_{split}.pqt').as_posix(), engine='pyarrow', index=True)
        results.to_parquet((self.run_dir / f'results_{split}.pqt').as_posix(), engine='pyarrow', index=True)
