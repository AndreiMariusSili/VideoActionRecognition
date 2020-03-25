import abc
import copy
import dataclasses as dc
import glob
import json
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
import torch.nn as nn
import torch.nn.functional as nn_func

import constants as ct
import databunch.databunch as db
import databunch.dataset as ds
import databunch.label as dl
import databunch.video as dv
import helpers as hp
import logger as lg
import models.tadn.base.base_tadn as tadn
import models.tarn.common.temporal_encoder as tarn
import options.experiment_options as eo
import specs.maps as sm


def flow2img(flow_data):
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    unknow_flow_threshold = 1e7
    pr1 = abs(u) > unknow_flow_threshold
    pr2 = abs(v) > unknow_flow_threshold
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    height, width = u.shape
    img = np.zeros((height, width, 3))

    nan_idx = np.isnan(u) | np.isnan(v)
    u[nan_idx] = v[nan_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nan_idx)))

    return img


def make_color_wheel():
    ry = 15
    yg = 6
    GC = 4
    cb = 11
    bm = 13
    mr = 6

    ncols = ry + yg + GC + cb + bm + mr

    color_wheel = np.zeros([ncols, 3])

    col = 0

    # RY
    color_wheel[0:ry, 0] = 255
    color_wheel[0:ry, 1] = np.transpose(np.floor(255 * np.arange(0, ry) / ry))
    col += ry

    # YG
    color_wheel[col:col + yg, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, yg) / yg))
    color_wheel[col:col + yg, 1] = 255
    col += yg

    # GC
    color_wheel[col:col + GC, 1] = 255
    color_wheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    color_wheel[col:col + cb, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cb) / cb))
    color_wheel[col:col + cb, 2] = 255
    col += cb

    # BM
    color_wheel[col:col + bm, 2] = 255
    color_wheel[col:col + bm, 0] = np.transpose(np.floor(255 * np.arange(0, bm) / bm))
    col += + bm

    # MR
    color_wheel[col:col + mr, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, mr) / mr))
    color_wheel[col:col + mr, 0] = 255

    return color_wheel


def recon2frames(recon: np.ndarray, flow: bool = False) -> t.List[np.ndarray]:
    _t, h, w, c = recon.shape

    if flow:
        frames = [flow2img(flow.squeeze()) for flow in np.split(recon, _t)]
    else:
        frames = [frame.squeeze() for frame in np.split(recon, _t)]

    return frames


def plot_frames(frames: t.List[np.ndarray], titles: t.List[str] = None, cols: int = 1,
                group: bool = True) -> t.Union[plt.Figure, t.List[plt.Figure]]:
    assert ((titles is None) or (len(frames) == len(titles)))
    num_frames = len(frames)
    if titles is None:
        titles = [f'Frame {i:2d}' % i for i in range(1, num_frames + 1)]
    if group:
        fig: plt.Figure = plt.figure()
        for n, (image, title) in enumerate(zip(frames, titles)):
            a = fig.add_subplot(np.ceil(num_frames / float(cols)), cols, n + 1)
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * num_frames)
        [ax.get_xaxis().set_visible(False) for ax in fig.axes]
        [ax.get_yaxis().set_visible(False) for ax in fig.axes]
        fig.tight_layout()
        fig.show()

        return fig
    else:
        figs = []
        for n, (image, title) in enumerate(zip(frames, titles)):
            fig: plt.Figure = plt.figure()
            [ax.get_xaxis().set_visible(False) for ax in fig.axes]
            [ax.get_yaxis().set_visible(False) for ax in fig.axes]
            plt.imshow(image)
            plt.axis('off')
            fig.tight_layout()
            fig.show()
            figs.append(fig)

        return figs


def plot_class_heat_map(class_values: np.ndarray, title: str = None) -> plt.Figure:
    fig: plt.Figure = plt.figure()
    plt.imshow(class_values, cmap='gray')
    if title:
        plt.title(title)
    fig.tight_layout()
    [ax.get_xaxis().set_visible(False) for ax in fig.axes]
    [ax.get_yaxis().set_visible(False) for ax in fig.axes]
    fig.show()

    return fig


class BaseVisualiser(abc.ABC):
    def __init__(self, opts: t.Optional[eo.ExperimentOptions]):
        self.opts = opts
        self.general_viz_dir = ct.WORK_ROOT / 'figures'
        self.general_viz_dir.mkdir(exist_ok=True)

        if self.opts:
            self.run_viz_dir = ct.WORK_ROOT / self.opts.run_dir / 'figures'
            self.run_viz_dir.mkdir(exist_ok=True)
            self.device = th.device(f'cpu')
            self.data_bunch, self.opts.model.opts.num_classes = self._init_databunch()
            self.best_ckpt = self._get_best_ckpt()
            self.model = self._init_model()
            self.logger = lg.ExperimentLogger(self.opts, True, self.opts.trainer.metrics, self.opts.evaluator.metrics)
            opts = dc.asdict(self.opts)
            hp.path_to_string(opts)
            self.logger.log(json.dumps(opts, indent=True))

            th.set_grad_enabled(False)

    def _get_best_ckpt(self) -> str:
        best_models = list(glob.glob((ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'best_model_*.pth').as_posix()))
        if len(best_models) > 1:
            raise ValueError('More than one best model available. Remove old versions.')
        return best_models.pop()

    def _init_model(self) -> nn.Module:
        opts = dc.asdict(copy.deepcopy(self.opts.model.opts))
        del opts['batch_size']
        model = sm.Models[self.opts.model.arch].value(**opts).to(self.device)
        model.load_state_dict(th.load(self.best_ckpt, map_location=self.device))

        model = model.eval()

        return model

    def _init_databunch(self) -> t.Tuple[db.VideoDataBunch, int]:
        self.opts.databunch.train_dso.read_jpeg = False
        self.opts.databunch.dev_dso.read_jpeg = False
        self.opts.databunch.test_dso.read_jpeg = False
        self.opts.databunch.train_dso.setting = 'eval'
        self.opts.databunch.dev_dso.setting = 'eval'
        self.opts.databunch.test_dso.setting = 'eval'
        self.opts.databunch.dlo.shuffle = False
        self.opts.databunch.dlo.batch_size *= self.opts.world_size
        self.opts.databunch.distributed = False
        data_bunch = db.VideoDataBunch(db_opts=self.opts.databunch)

        return data_bunch, len(data_bunch.lids)

    def plot_input(self, split: str, ds_idx: int, save: bool = True, grp: bool = True) -> t.List[plt.Figure]:
        dataset = self.select_dataset(split)
        video, label = self.get_data_point(split, ds_idx)

        length = dataset.so.num_segments
        frames = recon2frames(video.data, False)
        titles = [f't={_t}' for _t in range(length)]

        fig_or_figs = plot_frames(frames, titles, len(frames), grp)
        if isinstance(fig_or_figs, plt.Figure):
            self.save_fig(fig_or_figs, f'frames_target_{split}_{ds_idx}.jpg', save)
            return [fig_or_figs]
        elif isinstance(fig_or_figs, list):
            for i, fig in enumerate(fig_or_figs):
                self.save_fig(fig, f'frames_target_{split}_{ds_idx}_{i}.jpg', save)
            return fig_or_figs

    def plot_class_target(self, split: str, dataset_idx: int, save: bool = True) -> t.List[plt.Figure]:
        dataset = self.select_dataset(split)
        video, label = self.get_data_point(split, dataset_idx)

        target = th.tensor(label.data, dtype=th.int64, device=self.device).reshape(-1)
        one_hot_target = th.zeros(len(dataset.lids), dtype=th.int64, device=self.device)
        one_hot_target = one_hot_target.scatter(0, target, 1).reshape(-1, 1).cpu().numpy()

        fig = plot_class_heat_map(one_hot_target, label.meta.label)
        self.save_fig(fig, f'class_target_{split}_{dataset_idx}.jpg', save)

        return [fig]

    def plot_class_pred(self, split: str, dataset_idx: int, save: bool = True) -> t.List[plt.Figure]:
        dataset = self.select_dataset(split)
        video, label = self.get_data_point(split, dataset_idx)

        figs = []
        preds = self.preds(video)
        for sample_idx, pred in enumerate(preds):
            lid = int(th.argmax(pred, dim=-1))
            pred = pred.reshape(-1, 1).cpu().numpy()
            label = dataset.lid2labels[lid]
            fig = plot_class_heat_map(pred, label)
            self.save_fig(fig, f'class_pred_{split}_{dataset_idx}_{sample_idx}.jpg', save)
            figs.append(fig)

        return figs

    def plot_class_loss(self, split: str, dataset_idx: int, save: bool = True) -> t.List[plt.Figure]:
        dataset = self.select_dataset(split)
        video, label = self.get_data_point(split, dataset_idx)

        figs = []
        target = th.tensor(label.data, dtype=th.int64, device=self.device).reshape(-1)
        preds = self.preds(video)
        for sample_idx, pred in enumerate(preds):
            loss = nn_func.cross_entropy(pred.unsqueeze(0), target)
            one_hot_loss = th.zeros(len(dataset.lids), dtype=th.int64, device=self.device)
            one_hot_loss = one_hot_loss.scatter(0, target, loss).reshape(-1, 1).cpu().numpy()
            fig = plot_class_heat_map(one_hot_loss)
            self.save_fig(fig, f'class_loss_{split}_{dataset_idx}_{sample_idx}.jpg', save)
            figs.append(fig)

        return figs

    def plot_recon_target(self, split: str, ds_idx: int, save: bool = True, grp: bool = True) -> t.List[plt.Figure]:
        video, label = self.get_data_point(split, ds_idx)

        frames = recon2frames(video.recon, self.opts.model.opts.flow)
        titles = [f't={_t}' for _t in range(len(frames))]

        fig_or_figs = plot_frames(frames, titles, len(frames), grp)
        if isinstance(fig_or_figs, plt.Figure):
            self.save_fig(fig_or_figs, f'frames_target_{split}_{ds_idx}.jpg', save)
            return [fig_or_figs]
        elif isinstance(fig_or_figs, list):
            for i, fig in enumerate(fig_or_figs):
                self.save_fig(fig, f'frames_target_{split}_{ds_idx}_{i}.jpg', save)
            return fig_or_figs

    def plot_recon_pred(self, split: str, ds_idx: int, save: bool = True, grp: bool = True) -> t.List[plt.Figure]:
        video, label = self.get_data_point(split, ds_idx)
        video.to_tensor()

        figs = []
        recons = self.recons(video)
        for sample_idx, recon in enumerate(recons):
            recon = np.transpose(recon.cpu().numpy(), axes=(0, 2, 3, 1))
            recon = recon2frames(recon, self.opts.model.opts.flow)
            titles = [f't={_t}' for _t in range(len(recon))]
            fig_or_figs = plot_frames(recon, titles, len(recon), grp)
            if isinstance(fig_or_figs, plt.Figure):
                self.save_fig(fig_or_figs, f'frames_recon_{split}_{ds_idx}_{sample_idx}.jpg', save)
            elif isinstance(fig_or_figs, list):
                for i, fig in enumerate(fig_or_figs):
                    self.save_fig(fig, f'frames_recon_{split}_{ds_idx}_{sample_idx}_{i}.jpg', save)
            figs.append(fig_or_figs)

        return figs

    def plot_recon_loss(self, split: str, dataset_idx: int, save: bool = True) -> t.List[plt.Figure]:
        dataset = self.select_dataset(split)
        video, label = self.get_data_point(split, dataset_idx)
        video_data = video.flow if self.opts.model.opts.flow else video.data

        figs = []
        length = dataset.so.num_segments
        titles = [f't={_t}' for _t in range(length)]
        target = th.tensor(video_data, dtype=th.float32, device=self.device).unsqueeze(0)
        recons = self.recons(video)
        for sample_idx, recon in enumerate(recons):
            loss = nn_func.smooth_l1_loss(recon.unsqueeze(0), target, reduction='none').squeeze()
            loss = (loss - th.min(loss)) / (th.max(loss) - th.min(loss))  # distribute to the full range [0, 1]
            loss = recon2frames(loss.numpy())
            fig = plot_frames(loss, titles, length)
            self.save_fig(fig, f'frames_loss_{split}_{dataset_idx}_{sample_idx}.jpg', save)
            figs.append(fig)

        return figs

    def plot_class_embeds(self, split: str, save: bool = True) -> plt.Figure:
        dataset = self.select_dataset(split)
        ids = np.load(ct.WORK_ROOT / self.opts.run_dir / split / 'tsne_ids.npy')
        class_embeds = np.load(ct.WORK_ROOT / self.opts.run_dir / split / 'class_embeds_tsne.npy')

        labels = dataset.meta.reindex(ids)[['label']].values
        class_embeds = pd.DataFrame(np.concatenate([class_embeds, labels], axis=1), columns=['x1', 'x2', 'label'])
        labels = pd.unique(class_embeds['label'])[0:12]
        class_embeds = class_embeds[class_embeds['label'].isin(labels)]
        sns.scatterplot(x='x1', y='x2', hue='label', data=class_embeds, legend=False)
        plt.show()
        self.save_fig(plt.gcf(), f'class_embeds_{split}.jpg', save)

        return plt.gcf()

    def plot_temporal_parameter_growth(self, save: bool = True):
        sizes = {
            'Time Steps': [],
            'DenseNet Temporal Encoder': [],
            'ResNet Temporal Encoder': []
        }

        for _t in range(4, 65, 4):
            tadn_size = hp.count_parameters(tadn.TemporalDenseNetEncoder(64, _t, 64, 0.5))
            tarn_size = hp.count_parameters(tarn.TemporalResNetEncoder(_t, 64))
            sizes['Time Steps'].append(_t)
            sizes['DenseNet Temporal Encoder'].append(tadn_size)
            sizes['ResNet Temporal Encoder'].append(tarn_size)

        sizes = pd.DataFrame \
            .from_dict(sizes) \
            .melt(id_vars='Time Steps', value_vars=['DenseNet Temporal Encoder', 'ResNet Temporal Encoder'],
                  var_name='Model', value_name='Parameters')
        fig = sns.lineplot(x='Time Steps', y='Parameters', hue='Model', data=sizes).get_figure()
        plt.show()
        self.save_fig(fig, f'temporal_parameter_growth.jpg', save)

        return fig

    def get_data_point(self, split: str, idx: int) -> t.Tuple[dv.Video, dl.Label]:
        dataset = self.select_dataset(split)
        video, label, meta = dataset[idx]
        video.to_numpy()
        label.to_numpy()

        return video, label

    def select_dataset(self, split: str) -> ds.VideoDataset:
        assert split in ['train', 'dev', 'test'], f'Unknown split: {split}.'
        if split == 'train':
            return self.data_bunch.train_set
        elif split == 'dev':
            return self.data_bunch.dev_set
        else:
            return self.data_bunch.test_set

    def save_fig(self, fig: plt.Figure, name: str, save: bool):
        _dir = self.run_viz_dir if self.opts else self.general_viz_dir
        if save:
            fig.savefig(_dir / name,
                        bbox_inches='tight',
                        dpi=120,
                        transparent=True)

    @abc.abstractmethod
    def recons(self, video: dv.Video) -> t.List[th.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def preds(self, video: dv.Video) -> t.List[th.Tensor]:
        raise NotImplementedError
