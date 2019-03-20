import json
from typing import List, Dict, Tuple, Any

import bokeh.models.widgets as bomw
import sklearn.metrics as skm
import bokeh.plotting as bop
import bokeh.palettes as pal
import bokeh.document as bod
import bokeh.layouts as bol
import bokeh.models as bom
import dataclasses as dc
from glob import glob
import pathlib as pl
from torch import nn
import pandas as pd
import numpy as np
import torch as th

import models.options as mo
import pipeline as pipe
import constants as ct
import helpers as hp


class Visualisation(object):
    page: str
    tabs: List[bomw.Panel]
    doc: bod.Document
    stats: bom.ColumnDataSource
    model: nn.Module
    run_dir: pl.Path
    run_opts: mo.RunOptions
    best_ckpt: str

    labels2id: pd.DataFrame
    id2labels: pd.DataFrame
    label2group: pd.DataFrame
    group_id2group_label: pd.DataFrame

    train_results: pd.DataFrame
    valid_results: pd.DataFrame

    videos: List[pipe.Video]
    labels: List[pipe.Label]
    predictions: List[pipe.Prediction]

    split: str
    train_label_confusion: bom.ColumnDataSource
    valid_label_confusion: bom.ColumnDataSource
    train_label_confusion_fig: bop.Figure
    valid_label_confusion_fig: bop.Figure
    train_group_confusion: bom.ColumnDataSource
    valid_group_confusion: bom.ColumnDataSource
    train_group_confusion_fig: bop.Figure
    valid_group_confusion_fig: bop.Figure

    def __init__(self, page: str, spec: mo.RunOptions):
        """Init a visualizer with a run model. They need to match."""
        self.page = page
        self.run_dir = ct.RUN_DIR / spec.name
        self.run_opts = spec
        self._get_best_ckpt()._load_data()._load_stats()._load_results()._setup_bokeh()

    def start(self):
        """start the visualisation assuming a bokeh server."""
        if self.page == 'details':
            self._start_details_page()
        elif self.page == 'stats':
            self._start_run_stats_page()
        elif self.page == 'results':
            self._start_results_page()

    def _get_best_ckpt(self) -> 'Visualisation':
        """Get the path to the best checkpoint."""
        model_paths = sorted(glob((self.run_dir / 'run_model_*.pth').as_posix()))
        scores = np.array(map(self._to_score, model_paths))
        best_idx = scores.argmin()
        self.best_ckpt = model_paths[best_idx]

        return self

    def _load_data(self) -> 'Visualisation':
        """Load a DataBunch and label / group id mappings."""
        self.data_bunch = self.run_opts.data_bunch(self.run_opts.data_bunch_opts,
                                                   self.run_opts.train_data_set_opts,
                                                   self.run_opts.valid_data_set_opts,
                                                   self.run_opts.train_data_loader_opts,
                                                   self.run_opts.valid_data_loader_opts)
        self.labels2id = hp.read_smth_labels2id(ct.SMTH_LABELS2ID)
        self.id2labels = hp.read_smth_id2labels(ct.SMTH_LABELS2ID)
        self.label2group = hp.read_smth_label2group(ct.SMTH_LABELS2GROUP)
        self.group_id2group_label = hp.read_smth_group2group_label(ct.SMTH_GROUP_ID2GROUP_LABEL)

        return self

    def _load_stats(self) -> 'Visualisation':
        """Load training/evaluation statistics."""
        stats = pd.read_csv(self.run_dir / 'stats.csv')
        stats['epoch'] = stats.index
        stats['train_diff'] = stats['train_acc@3'] - stats['train_acc@1']
        stats['valid_diff'] = stats['valid_acc@3'] - stats['valid_acc@1']
        self.stats = bom.ColumnDataSource(stats)

        return self

    def _load_results(self) -> 'Visualisation':
        self.train_results = hp.read_smth_results(self.run_dir / 'results_train.json')
        self.valid_results = hp.read_smth_results(self.run_dir / 'results_valid.json')

        return self

    def _setup_bokeh(self) -> 'Visualisation':
        """Setup the current document and other one-time setups of bokeh."""
        self.doc = bop.curdoc()
        self.tabs = []

        return self

    def _to_score(self, path: pl.Path) -> float:
        """Get the score of a checkpoint from its filename."""
        return float(path.as_posix().replace('.pth', '').split('=').pop())

    def _start_details_page(self):
        """Create a tab with a table of run options and a snippet of the data."""
        with open((self.run_dir / 'options.json').as_posix()) as file:
            details = json.load(file)
        dataset_opts = dict(**details['train_data_set_opts']['do'], **details['train_data_set_opts']['so'])
        tables = [
            self.__create_table_source(details),
            self.__create_table_source(details['model_opts']),
            self.__create_table_source(details['data_bunch_opts']),
            self.__create_table_source(details['train_data_loader_opts']),
            self.__create_table_source(dataset_opts),
            self.__create_table_source(details['trainer_opts']),
            self.__create_table_source(details['trainer_opts']['optimizer_opts'])
        ]
        for i, (source, columns) in enumerate(tables):
            table = bom.DataTable(source=source, columns=columns, header_row=True, fit_columns=False,
                                  reorderable=False, sortable=False, height=200)
            tables[i] = bol.widgetbox(table, sizing_mode='stretch_both')
        layout = bol.layout(
            children=[
                [tables[0], tables[1]],
                [tables[2], tables[3]],
                [tables[4], tables[5]],
                [bol.Spacer(), tables[6]]
            ], sizing_mode='stretch_both')
        self.doc.add_root(layout)

    def _start_run_stats_page(self):
        """Create a grid of training/validation statistics and appends them to the list of tabs."""
        fkwargs = {
            'height': 300,
            'sizing_mode': 'scale_width',
            'x_range': None,
            'y_range': None
        }
        lkwargs = {
            'line_width': 3
        }
        ys = list(zip(['train_loss', 'valid_loss'], ['Train Loss', 'Valid Loss']))
        loss_plot = self.__create_line_plot('Loss Over Epochs', 'Loss', 'epoch', ys, fkwargs, lkwargs)
        fkwargs['y_range'] = (0, 1)
        ys = list(zip(['train_acc@1', 'valid_acc@1'], ['Train Acc@1', 'Valid Acc@1']))
        top1_plot = self.__create_line_plot('Accuracy@1 Over Epochs', 'Acc@1', 'epoch', ys, fkwargs, lkwargs)
        ys = list(zip(['train_acc@3', 'valid_acc@3'], ['Train Acc@3', 'Valid Acc@3']))
        top3_plot = self.__create_line_plot('Accuracy@3 Over Epochs', 'Acc@3', 'epoch', ys, fkwargs, lkwargs)
        ys = list(zip(['train_diff', 'valid_diff'], ['Train Acc Diff', 'Valid Acc Diff']))
        diff_plot = self.__create_line_plot('Accuracy@3 - Accuracy@1 Over Epochs', 'Acc Diff', 'epoch', ys,
                                            fkwargs, lkwargs)

        grid = bol.gridplot([[top1_plot, top3_plot], [loss_plot, diff_plot]], sizing_mode='scale_width')
        self.doc.add_root(grid)

    def _start_results_page(self):
        """Show a batch of results and confusion matrices."""
        self.__sample_batch().__init_results_doc().__init_tick() \
            .__convert_to_rgba().__render_batch().__arrange_batch() \
            .__on('train') \
            .__create_label_confusion(False).__render_label_confusion(False) \
            .__create_group_confusion(False).__render_group_confusion(False) \
            .__on('valid') \
            .__create_label_confusion(False).__render_label_confusion(False) \
            .__create_group_confusion(False).__render_group_confusion(False) \
            .__render_results_page()

    def __on(self, split: str) -> 'Visualisation':
        assert split in ['train', 'valid'], f'Unknown split: {split}.'
        self.split = split

        return self

    def __sample_batch(self, n: int = 10) -> 'Visualisation':
        """Get a batch of videos from one of the validation split."""
        self.videos, self.labels = self.data_bunch.get_batch(n, 'valid')
        self.predictions = [pipe.Prediction(self.valid_results.loc[video.meta.id]) for video in self.videos]

        return self

    def __init_results_doc(self) -> 'Visualisation':
        """Load document CSS."""
        with open(ct.STYLES, 'r') as file:
            style = bom.Div(text=f'<style>{file.read()}</style>')
        self.doc.add_root(style)

        return self

    def __convert_to_rgba(self) -> 'Visualisation':
        """Convert videos to RGBA representation and flip for bokeh compatibility."""
        for i in range(len(self.videos)):
            # noinspection PyTypeChecker
            self.videos[i].data = [np.flipud(np.array(frame.convert('RGBA'))) for frame in self.videos[i].data]

        return self

    def __init_tick(self) -> 'Visualisation':
        """Initialize the step and data source."""
        self.step = 0
        self.source = bom.ColumnDataSource({video.meta.id: [video.data[self.step]] for video in self.videos})

        return self

    def __render_batch(self) -> 'Visualisation':
        """Create columns of image, ground truth and top3 predictions."""
        self.batch = []
        for video, label, prediction in zip(self.videos, self.labels, self.predictions):
            fig = bop.figure(x_range=(0, 290), y_range=(0, 224), height=224, width=280)
            video.show(fig, self.source)
            label.show(fig)
            pred_graph = prediction.show(self.id2labels, ['no-margin', 'scroll'])
            self.batch.append(bol.column([fig, pred_graph]))

        return self

    def __arrange_batch(self) -> 'Visualisation':
        """Arrange all figures in a grid layout."""
        no_rows = 2
        no_cols = len(self.batch) // 2 + len(self.batch) % 2
        rows = []
        for i in range(no_rows):
            row_figs = self.batch[(no_cols * i):(no_cols * (i + 1))]
            rows.append(bol.row(row_figs, height=300))
        self.batch = rows

        return self

    def __create_label_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Creates a stacked DataFrame representing the confusion matrix and loads it into a ColumnDataSource."""
        if self.split == 'train':
            results = self.train_results
        else:
            results = self.valid_results

        confusion = skm.confusion_matrix(results['template_id'].values, results['top1_pred'].values)
        if normalize:
            confusion = confusion.astype(np.float) / confusion.astype(np.float).sum(axis=1)[:, np.newaxis]
        confusion = pd.DataFrame(pd.DataFrame(confusion).stack(), columns=['Overlap']).reset_index()
        confusion.columns = ['Ground Truth', 'Prediction', 'Overlap']
        confusion = confusion.join(self.id2labels, on='Ground Truth') \
            .drop(labels=['Ground Truth', 'id'], axis=1) \
            .rename(columns={'template': 'Ground Truth'})
        confusion = confusion.join(self.id2labels, on='Prediction') \
            .drop(labels=['Prediction', 'id'], axis=1) \
            .rename(columns={'template': 'Prediction'})

        if self.split == 'train':
            self.train_label_confusion = bom.ColumnDataSource(confusion)
        else:
            self.valid_label_confusion = bom.ColumnDataSource(confusion)

        return self

    def __render_label_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Render the confusion matrix as a bokeh heat map."""
        if self.split == 'train':
            label_confusion = self.train_label_confusion
        else:
            label_confusion = self.valid_label_confusion

        if normalize:
            tooltips = [
                ('Overlap', '@Overlap{0.0000}'),
                ('Ground Truth', '@{Ground Truth}'),
                ('Prediction', '@{Prediction}')
            ]
        else:
            tooltips = [
                ('Overlap', '@Overlap'),
                ('Ground Truth', '@{Ground Truth}'),
                ('Prediction', '@{Prediction}')
            ]
        x_range = list(self.id2labels['template'])
        y_range = list(reversed(self.id2labels['template']))

        fig = bop.figure(title=f'{self.split.capitalize()} Class Confusion',
                         x_range=x_range, y_range=y_range,
                         x_axis_location="above", plot_width=712, plot_height=712,
                         tools='hover', toolbar_location='below', tooltips=tooltips)
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_text_font_size = '8pt'
        fig.axis.major_label_standoff = 0
        fig.xaxis.major_label_orientation = np.pi / 2.5
        fig.xaxis.axis_label = 'Prediction'
        fig.yaxis.axis_label = 'Ground Truth'

        if normalize:
            low, high = 0, 1
        else:
            low, high = label_confusion.data['Overlap'].min(), label_confusion.data['Overlap'].max()
        mapper = bom.LinearColorMapper(palette='Greys256', low=low, high=high)
        fig.rect(x="Prediction", y="Ground Truth", width=1, height=1, source=label_confusion,
                 fill_color={'field': 'Overlap', 'transform': mapper}, line_color=None)

        if self.split == 'train':
            self.train_label_confusion_fig = fig
        else:
            self.valid_label_confusion_fig = fig

        return self

    def __create_group_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Creates a stacked DataFrame representing the confusion matrix and loads it into a ColumnDataSource."""
        if self.split == 'train':
            results = self.train_results[['template_id', 'top3_pred_1', 'top3_pred_2', 'top3_pred_3']]
        else:
            results = self.valid_results[['template_id', 'top3_pred_1', 'top3_pred_2', 'top3_pred_3']]
        self.id2group = self.label2group.join(self.labels2id['id']).set_index('id', drop=True)
        group_results = results \
            .join(self.id2group['group'], on='template_id').rename(columns={'group': 'ground'}) \
            .join(self.id2group['group'], on='top3_pred_1').rename(columns={'group': 'group_1'}) \
            .join(self.id2group['group'], on='top3_pred_2').rename(columns={'group': 'group_2'}) \
            .join(self.id2group['group'], on='top3_pred_3').rename(columns={'group': 'group_3'})
        group_results = group_results.drop(labels=['template_id', 'top3_pred_1', 'top3_pred_2', 'top3_pred_3'], axis=1)
        group_results = group_results.melt(['ground'], ['group_1', 'group_2', 'group_3'], None, 'pred')
        confusion = skm.confusion_matrix(group_results['ground'].values, group_results['pred'].values)
        if normalize:
            confusion = confusion.astype(np.float) / confusion.astype(np.float).sum(axis=1)[:, np.newaxis]

        confusion = pd.DataFrame(confusion).stack().reset_index()
        confusion.columns = ['Ground Truth', 'Prediction', 'Overlap']
        confusion = confusion \
            .join(self.group_id2group_label['label'], on='Ground Truth') \
            .drop(labels=['Ground Truth'], axis=1) \
            .rename(columns={'label': 'Ground Truth'})
        confusion = confusion \
            .join(self.group_id2group_label['label'], on='Prediction') \
            .drop(labels=['Prediction'], axis=1) \
            .rename(columns={'label': 'Prediction'})

        if self.split == 'train':
            self.train_group_confusion = bom.ColumnDataSource(confusion)
        else:
            self.valid_group_confusion = bom.ColumnDataSource(confusion)

        return self

    def __render_group_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Render the confusion matrix as a bokeh heat map."""
        if self.split == 'train':
            group_confusion = self.train_group_confusion
        else:
            group_confusion = self.valid_group_confusion
        if normalize:
            tooltips = [
                ('Overlap', '@Overlap{0.0000}'),
                ('Ground Truth', '@{Ground Truth}'),
                ('Prediction', '@Prediction')
            ]
        else:
            tooltips = [
                ('Overlap', '@Overlap'),
                ('Ground Truth', '@{Ground Truth}'),
                ('Prediction', '@Prediction')
            ]
        x_range = list(self.group_id2group_label['label'].map(str).unique().tolist())
        y_range = list(reversed(self.group_id2group_label['label'].map(str).unique().tolist()))

        fig = bop.figure(title=f'{self.split.capitalize()} Super Class Confusion',
                         x_range=x_range, y_range=y_range,
                         x_axis_location="above", plot_width=712, plot_height=712,
                         tools='hover', toolbar_location='below', tooltips=tooltips)
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_text_font_size = "8pt"
        fig.axis.major_label_standoff = 0
        fig.xaxis.major_label_orientation = np.pi / 2.5
        fig.xaxis.axis_label = 'Prediction'
        fig.yaxis.axis_label = 'Ground Truth'

        if normalize:
            low, high = 0, 1
        else:
            low, high = group_confusion.data['Overlap'].min(), group_confusion.data['Overlap'].max()
        mapper = bom.LinearColorMapper(palette='Greys256', low=low, high=high)
        fig.rect(x="Prediction", y="Ground Truth", width=1, height=1, source=group_confusion,
                 fill_color={'field': 'Overlap', 'transform': mapper}, line_color=None)

        if self.split == 'train':
            self.train_group_confusion_fig = fig
        else:
            self.valid_group_confusion_fig = fig

        return self

    def __render_results_page(self):
        """Render batch and confusions matrices in the DOM. Add a periodic callback to play videos."""
        self.doc.add_root(bol.layout(self.batch, sizing_mode='fixed'))
        self.doc.add_root(
            bol.layout([[self.valid_label_confusion_fig, self.valid_group_confusion_fig]], sizing_mode='fixed'))
        self.doc.add_root(
            bol.layout([[self.train_label_confusion_fig, self.train_group_confusion_fig]], sizing_mode='fixed'))
        self.doc.add_periodic_callback(self.__tick, 84)

    def __tick(self) -> None:
        """Increment step and update data source."""
        self.step += 1
        self.source.data = {video.meta.id: [video.data[self.step % video.meta.length]] for video in self.videos}

    def __create_table_source(self, data: Dict[str, Any]) -> Tuple[bom.ColumnDataSource, List[bom.TableColumn]]:
        """Create a table with 2 columns: Option, Value."""
        source = dict(opt=[], val=[])
        cols = [bom.TableColumn(field='opt', title='Option'), bom.TableColumn(field='val', title='Value')]
        for opt, val in data.items():
            if val is None:
                continue
            if isinstance(val, dict):
                continue
            if isinstance(val, str) and val.startswith('<class'):
                val = val.replace("'>", "").split('.')[-1]
            source['opt'].append(opt)
            source['val'].append(val)
        source = bom.ColumnDataSource(source)

        return source, cols

    def __create_line_plot(self, title: str, tip: str, x: str, ys: List[Tuple[str, str]],
                           fkwargs: Dict[str, any], lkwargs: Dict[str, any]) -> bom.Plot:
        """Create a line plot with hover tool from a stats DataFrame."""
        hover = bom.HoverTool(tooltips=[
            ('Epoch', '@epoch{0,}'),
            (f'Train {tip}', '@$name{0.0000}'),
            (f'Valid {tip}', '@$name{0.0000}'),
        ], mode='vline')
        plot = bop.figure(title=title, tools=[hover, 'box_zoom', 'wheel_zoom', 'pan', 'reset'], **fkwargs)
        for i, (y, leg) in enumerate(ys):
            # noinspection PyUnresolvedReferences
            plot.line(x=x, y=y, source=self.stats, name=y, color=pal.Spectral6[i % len(ys)], legend=leg, **lkwargs)

        return plot
