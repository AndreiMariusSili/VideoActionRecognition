import json
import pathlib as pl
from glob import glob
from typing import Any, Dict, List, Tuple

import bokeh.document as bod
import bokeh.layouts as bol
import bokeh.models as bom
import bokeh.palettes as pal
import bokeh.plotting as bop
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from torch import nn

import constants as ct
import databunch.databunch as db
import helpers as hp
import options.experiment_options
from databunch import label as pil, prediction as pip, video as piv


class Visualisation(object):
    page: str
    mode: str
    doc: bod.Document
    stats: bom.ColumnDataSource
    model: nn.Module
    cut: str
    run_dir: pl.Path
    run_opts: options.experiment_options.ExperimentOptions
    best_ckpt: str

    lid2label: pd.DataFrame
    label2lid: pd.DataFrame
    label2gid: pd.DataFrame

    train_results: pd.DataFrame
    valid_results: pd.DataFrame
    test_results: pd.DataFrame

    videos: List[piv.Video]
    labels: List[pil.Label]
    predictions: List[pip.Prediction]

    split: str
    train_label_confusion: bom.ColumnDataSource
    valid_label_confusion: bom.ColumnDataSource
    train_label_confusion_fig: bop.Figure
    valid_label_confusion_fig: bop.Figure
    train_group_confusion: bom.ColumnDataSource
    valid_group_confusion: bom.ColumnDataSource
    train_group_confusion_fig: bop.Figure
    valid_group_confusion_fig: bop.Figure

    def __init__(self, page: str, opts: options.experiment_options.ExperimentOptions):
        """Init a visualizer with a run model. They need to match."""
        self.page = page
        self.mode = opts.mode
        self.name = opts.name
        self.cut = f'__{self.name.split("_").pop()}'
        self.run_dir = opts.run_dir / self.cut / self.name

        self.run_opts = opts
        self._load_data()._load_stats()._load_results()._setup_bokeh()

    def start(self):
        """start the visualisation assuming a bokeh server."""
        if self.page == 'details':
            self._start_details_page()
        elif self.page == 'stats':
            self._start_stats_page()
        elif self.page == 'results':
            self._start_results_page()
        elif self.page == 'confusion':
            self._start_confusion_page()
        elif self.page == 'embeddings':
            self._start_embeddings_page()

    def _get_best_ckpt(self) -> 'Visualisation':
        """Get the path to the best checkpoint."""
        model_paths = sorted(glob((self.run_dir / 'best_model_*.pth').as_posix()))
        scores = np.array(map(self._to_score, model_paths))
        best_idx = scores.argmin()
        self.best_ckpt = model_paths[best_idx]

        return self

    def _load_data(self) -> 'Visualisation':
        """Load a DataBunch and label / group id mappings."""
        self.data_bunch = db.VideoDataBunch(self.run_opts.dbo,
                                            self.run_opts.train_dso,
                                            self.run_opts.dev_dso,
                                            self.run_opts.test_dso,
                                            self.run_opts.train_dlo,
                                            self.run_opts.dev_dlo,
                                            self.run_opts.test_dlo)
        self.label2lid = hp.read_label2lid()
        self.lid2label = hp.read_smth_lid2label()
        self.label2gid = hp.read_smth_label2gid()
        self.gid2labels = hp.read_gid2labels()

        return self

    def _load_stats(self) -> 'Visualisation':
        """Load training/evaluation statistics."""
        stats = pd.read_csv(self.run_dir / 'training_stats.csv')
        stats['epoch'] = stats.index
        self.stats = bom.ColumnDataSource(stats)

        return self

    def _load_results(self) -> 'Visualisation':
        try:
            self.train_results = hp.read_results(self.run_dir / 'results_train.tar')
        except FileNotFoundError:
            pass
        try:
            self.dev_results = hp.read_results(self.run_dir / 'results_dev.tar')
        except FileNotFoundError:
            pass
        try:
            self.valid_results = hp.read_results(self.run_dir / 'results_valid.tar')
        except FileNotFoundError:
            pass

        return self

    def _setup_bokeh(self) -> 'Visualisation':
        """Setup the current document and other one-time setups of bokeh."""
        self.doc = bop.curdoc()

        return self

    def _to_score(self, path: pl.Path) -> float:
        """Get the score of a checkpoint from its filename."""
        return float(path.as_posix().replace('.pth', '').split('=').pop())

    def _start_details_page(self):
        """Create a tab with a table of run options and a snippet of the data."""
        with open((self.run_dir / 'options.json').as_posix()) as file:
            details = json.load(file)
        train_ds_opts = dict(**details['train_ds_opts']['do'], **details['train_ds_opts']['so'])
        dev_ds_opts = dict(**details['dev_ds_opts']['do'], **details['dev_ds_opts']['so'])
        valid_ds_opts = dict(**details['valid_ds_opts']['do'], **details['valid_ds_opts']['so'])
        tables = [
            self.__create_table_source(details, 'Run Options'),
            self.__create_table_source(details['db_opts'], 'DataBunch'),
            self.__create_table_source(details['model_opts'], 'Model'),
            self.__create_table_source(details['trainer_opts']['optimizer_opts'], 'Optimizer'),
            self.__create_table_source(train_ds_opts, 'Train Dataset'),
            self.__create_table_source(details['train_dl_opts'], 'Train DataLoader'),
            self.__create_table_source(dev_ds_opts, 'Dev Dataset'),
            self.__create_table_source(details['dev_dl_opts'], 'Dev DataLoader'),
            self.__create_table_source(valid_ds_opts, 'Valid Dataset'),
            self.__create_table_source(details['valid_dl_opts'], 'Valid DataLoader'),
            self.__create_table_source(details['trainer_opts'], 'Trainer'),
            self.__create_table_source(details['evaluator_opts'], 'Evaluator'),
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
                [tables[6], tables[7]],
                [tables[8], tables[9]],
                [tables[10], tables[11]]
            ], sizing_mode='stretch_both')
        self.doc.add_root(layout)

    def _start_stats_page(self):
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
        if self.run_opts.mode == 'class':
            ys = list(zip(['train_ce_loss', 'dev_ce_loss'], ['Train CE Loss', 'Dev CE Loss']))
            loss_plot = self.__create_line_plot('CE Loss Over Epochs', 'Loss', 'epoch', ys, fkwargs, lkwargs)
            fkwargs['y_range'] = (0, 1)
            ys = list(zip(['train_acc@1', 'dev_acc@1'], ['Train Acc@1', 'Dev Acc@1']))
            top1_plot = self.__create_line_plot('Accuracy@1 Over Epochs', 'Acc@1', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_acc@5', 'dev_acc@5'], ['Train Acc@5', 'Dev Acc@5']))
            top2_plot = self.__create_line_plot('Accuracy@5 Over Epochs', 'Acc@5', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_iou', 'dev_iou'], ['Train IoU', 'Dev IoU']))
            iou_plot = self.__create_line_plot('IoU Over Epochs', 'Acc Diff', 'epoch', ys, fkwargs, lkwargs)
            grid = bol.gridplot([
                [top1_plot, top2_plot, iou_plot],
                [loss_plot, bom.Spacer(), bom.Spacer()]
            ], sizing_mode='scale_width')
        elif self.run_opts.mode == 'ae':
            ys = list(zip(['train_ce_loss', 'dev_ce_loss'], ['Train CE Loss', 'Dev CE Loss']))
            ce_loss_plot = self.__create_line_plot('CE Loss Over Epochs', 'CE Loss', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_mse_loss', 'dev_mse_loss'], ['Train MSE Loss', 'Dev MSE Loss']))
            mse_loss_plot = self.__create_line_plot('MSE Loss Over Epochs', 'MSE Loss', 'epoch', ys, fkwargs, lkwargs)
            fkwargs['y_range'] = (0, 1)
            ys = list(zip(['train_acc@1', 'dev_acc@1'], ['Train Acc@1', 'Dev Acc@1']))
            top1_plot = self.__create_line_plot('Accuracy@1 Over Epochs', 'Acc@1', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_acc@5', 'dev_acc@5'], ['Train Acc@5', 'Dev Acc@5']))
            top2_plot = self.__create_line_plot('Accuracy@5 Over Epochs', 'Acc@5', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_iou', 'dev_iou'], ['Train IoU', 'Dev IoU']))
            iou_plot = self.__create_line_plot('IoU Over Epochs', 'Acc Diff', 'epoch', ys, fkwargs, lkwargs)
            grid = bol.gridplot([
                [top1_plot, top2_plot, iou_plot],
                [mse_loss_plot, ce_loss_plot, bom.Spacer()],
            ], sizing_mode='scale_width')
        else:
            ys = list(zip(['train_ce_loss', 'dev_ce_loss'], ['Train CE Loss', 'Dev CE Loss']))
            ce_loss_plot = self.__create_line_plot('CE Loss Over Epochs', 'CE Loss', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_mse_loss', 'dev_mse_loss'], ['Train MSE Loss', 'Dev MSE Loss']))
            mse_loss_plot = self.__create_line_plot('MSE Loss Over Epochs', 'MSE Loss', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_kld_loss', 'dev_kld_loss'], ['Train KLD Loss', 'Dev KLD Loss']))
            kld_loss_plot = self.__create_line_plot('KLD Loss Over Epochs', 'KLD Loss', 'epoch', ys, fkwargs, lkwargs)
            fkwargs['y_range'] = (0, 1)
            ys = list(zip(['train_acc@1', 'dev_acc@1'], ['Train Acc@1', 'Dev Acc@1']))
            top1_plot = self.__create_line_plot('Accuracy@1 Over Epochs', 'Acc@1', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_acc@5', 'dev_acc@5'], ['Train Acc@5', 'Dev Acc@5']))
            top2_plot = self.__create_line_plot('Accuracy@5 Over Epochs', 'Acc@5', 'epoch', ys, fkwargs, lkwargs)
            ys = list(zip(['train_iou', 'dev_iou'], ['Train IoU', 'Dev IoU']))
            iou_plot = self.__create_line_plot('IoU Over Epochs', 'Acc Diff', 'epoch', ys, fkwargs, lkwargs)
            grid = bol.gridplot([
                [top1_plot, top2_plot, iou_plot],
                [mse_loss_plot, ce_loss_plot, kld_loss_plot],
            ], sizing_mode='scale_width')

        self.doc.add_root(grid)

    def _start_results_page(self):
        """Show a batch of results and confusion matrices."""
        self.__sample_batch(18).__convert_to_rgba().__init_results_doc() \
            .__init_tick().__render_batch().__arrange_batch() \
            .__render_results_page()

    def _start_embeddings_page(self):
        self.__render_embeddings().__render_embeddings_page()

    def _start_confusion_page(self):
        self.__on('train') \
            .__create_label_confusion().__render_label_confusion() \
            .__create_group_confusion().__render_group_confusion() \
            .__on('dev') \
            .__create_label_confusion().__render_label_confusion() \
            .__create_group_confusion().__render_group_confusion() \
            .__on('valid') \
            .__create_label_confusion().__render_label_confusion() \
            .__create_group_confusion().__render_group_confusion() \
            .__render_confusion_page()

    def __on(self, split: str) -> 'Visualisation':
        assert split in ['train', 'dev', 'valid'], f'Unknown split: {split}.'
        self.split = split

        return self

    def __sample_batch(self, n: int = 10) -> 'Visualisation':
        """Get a batch of videos from one of the validation split."""
        self.videos, self.labels = self.data_bunch.get_batch(n, 'valid')
        self.predictions = [pip.Prediction(self.valid_results.loc[video.meta.id]) for video in self.videos]

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

    def __arrange_batch(self) -> 'Visualisation':
        """Arrange all figures in a grid layout."""
        no_cols = 6
        no_rows = len(self.batch) // no_cols + len(self.batch) % no_cols
        rows = []
        for i in range(no_rows):
            row_figs = self.batch[(no_cols * i):(no_cols * (i + 1))]
            rows.append(bol.row(row_figs, height=300))
        self.batch = rows

        return self

    def __render_batch(self) -> 'Visualisation':
        """Create columns of image, ground truth and top2 predictions."""
        self.batch = []
        for video, label, prediction in zip(self.videos, self.labels, self.predictions):
            fig = bop.figure(x_range=(0, 336), y_range=(0, 224), height=224, width=336)
            video.show(fig, self.source)
            label.show(fig)
            pred = prediction.show(self.lid2label, ['no-margin', 'scroll'])
            self.batch.append(bol.column([fig, pred], height=224, width=336))

        return self

    def __render_embeddings(self):
        """Render the plots of TSNE embeddings for train, dev, valid. Only show one legend since all the same."""
        train_embeddings = self.train_results[['proj_x1', 'proj_x2', 'template']]
        dev_embeddings = self.dev_results[['proj_x1', 'proj_x2', 'template']]
        valid_embeddings = self.valid_results[['proj_x1', 'proj_x2', 'template']]

        tools = ['hover', 'save']
        tooltips = [
            ('Template', '@template'),
        ]

        # legend_items = []
        fig = bop.figure(title='Train Embeddings Projection', tooltips=tooltips, tools=tools)
        for i, template in enumerate(sorted(self.train_results.template.unique())):
            # noinspection PyUnresolvedReferences
            df = train_embeddings[train_embeddings['template'] == template]
            df = df[df.proj_x1.notnull() & df.proj_x2.notnull()]
            source = bom.ColumnDataSource(df)
            fig.circle('proj_x1', 'proj_x2', source=source, size=7,
                       fill_color=pal.Category20_20[i],
                       line_color=None)
            # legend_items.append((template, [glyph]))
            if i == 19:
                break
        # legend = bom.Legend(items=legend_items, location='center')
        # fig.add_layout(legend, 'below')
        self.train_embeddings_figure = fig

        # legend_items = []
        fig = bop.figure(title='Dev Embeddings Projection', tooltips=tooltips, tools=tools)
        for i, template in enumerate(sorted(self.dev_results.template.unique())):
            # noinspection PyUnresolvedReferences
            source = bom.ColumnDataSource(dev_embeddings[dev_embeddings['template'] == template])
            fig.circle('proj_x1', 'proj_x2', source=source, size=7, fill_color=pal.Category20_20[i],
                       line_color=None)
            # legend_items.append((template, [glyph]))
            if i == 19:
                break
        # legend = bom.Legend(items=legend_items, location='center')
        # fig.add_layout(legend, 'below')
        self.dev_embeddings_figure = fig

        # legend_items = []
        fig = bop.figure(title='Valid Embeddings Projection', tooltips=tooltips, tools=tools)
        for i, template in enumerate(sorted(self.valid_results.template.unique())):
            # noinspection PyUnresolvedReferences
            source = bom.ColumnDataSource(valid_embeddings[valid_embeddings['template'] == template])
            fig.circle('proj_x1', 'proj_x2', source=source, size=7, fill_color=pal.Category20_20[i],
                       line_color=None)
            # legend_items.append((template, [glyph]))
            if i == 19:
                break

        # legend = bom.Legend(items=legend_items, location='center')
        # fig.add_layout(legend, 'below')
        self.valid_embeddings_figure = fig

        return self

    def __render_embeddings_page(self):
        row = [self.train_embeddings_figure, self.dev_embeddings_figure, self.valid_embeddings_figure]
        self.doc.add_root(bol.gridplot([row], sizing_mode='scale_width'))

    def __create_label_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Creates a stacked DataFrame representing the confusion matrix and loads it into a ColumnDataSource."""
        if self.split == 'train':
            results = self.train_results
        elif self.split == 'dev':
            results = self.dev_results
        else:
            results = self.valid_results

        confusion = skm.confusion_matrix(results['template_id'].values.astype(np.int),
                                         results['top1_pred'].values.astype(np.int))
        if normalize:
            confusion = confusion.astype(np.float) / confusion.astype(np.float).sum(axis=1)[:, np.newaxis]
        confusion = pd.DataFrame(confusion).stack().reset_index()
        confusion.columns = ['Ground Truth', 'Prediction', 'Overlap']
        confusion = confusion.join(self.lid2label, on='Ground Truth') \
            .drop(labels=['Ground Truth', 'id'], axis=1) \
            .rename(columns={'template': 'Ground Truth'})
        confusion = confusion.join(self.lid2label, on='Prediction') \
            .drop(labels=['Prediction', 'id'], axis=1) \
            .rename(columns={'template': 'Prediction'})

        if self.split == 'train':
            self.train_label_confusion = bom.ColumnDataSource(confusion)
        elif self.split == 'dev':
            self.dev_label_confusion = bom.ColumnDataSource(confusion)
        else:
            self.valid_label_confusion = bom.ColumnDataSource(confusion)

        return self

    def __render_label_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Render the confusion matrix as a bokeh heat map."""
        if self.split == 'train':
            label_confusion = self.train_label_confusion
        elif self.split == 'dev':
            label_confusion = self.dev_label_confusion
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
        x_range = list(self.lid2label['template'])
        y_range = list(reversed(self.lid2label['template']))

        fig = bop.figure(title=f'{self.split.capitalize()} Class Confusion',
                         x_range=x_range, y_range=y_range,
                         x_axis_location="above", width=712, height=712,
                         tools='hover,save', toolbar_location='below', tooltips=tooltips)
        fig.title.vertical_align = 'top'
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_text_font_size = '8pt'
        fig.axis.major_label_standoff = 0
        fig.xaxis.major_label_orientation = np.pi / 2
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
        elif self.split == 'dev':
            self.dev_label_confusion_fig = fig
        else:
            self.valid_label_confusion_fig = fig

        return self

    def __create_group_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Creates a stacked DataFrame representing the confusion matrix and loads it into a ColumnDataSource."""
        if self.split == 'train':
            results = self.train_results[['template_id', 'top2_pred_1', 'top2_pred_2']]
        elif self.split == 'dev':
            results = self.dev_results[['template_id', 'top2_pred_1', 'top2_pred_2']]
        else:
            results = self.valid_results[['template_id', 'top2_pred_1', 'top2_pred_2']]
        self.lid2gid = self.label2gid[['id']] \
            .join(self.label2lid['id'], how='inner', lsuffix='_group', rsuffix='_label') \
            .set_index('id_label', drop=True)
        group_results = results \
            .join(self.lid2gid['id_group'], on='template_id').rename(columns={'id_group': 'ground'}) \
            .join(self.lid2gid['id_group'], on='top2_pred_1').rename(columns={'id_group': 'group_1'}) \
            .join(self.lid2gid['id_group'], on='top2_pred_2').rename(columns={'id_group': 'group_2'})
        group_results = group_results.drop(labels=['template_id', 'top2_pred_1', 'top2_pred_2'], axis=1)
        group_results = group_results.melt(['ground'], ['group_1', 'group_2'], None, 'pred').drop('variable',
                                                                                                  axis=1)
        confusion = skm.confusion_matrix(group_results['ground'].values, group_results['pred'].values)
        if normalize:
            confusion = confusion.astype(np.float) / confusion.astype(np.float).sum(axis=1)[:, np.newaxis]

        confusion = pd.DataFrame(confusion).stack().reset_index()
        confusion.columns = ['Ground Truth', 'Prediction', 'Overlap']
        confusion['Ground Truth'] = confusion['Ground Truth'].map(str)
        confusion['Prediction'] = confusion['Prediction'].map(str)

        if self.split == 'train':
            self.train_group_confusion = bom.ColumnDataSource(confusion)
        elif self.split == 'dev':
            self.dev_group_confusion = bom.ColumnDataSource(confusion)
        else:
            self.valid_group_confusion = bom.ColumnDataSource(confusion)

        return self

    def __render_group_confusion(self, normalize: bool = True) -> 'Visualisation':
        """Render the confusion matrix as a bokeh heat map."""
        if self.split == 'train':
            group_confusion = self.train_group_confusion
        elif self.split == 'dev':
            group_confusion = self.dev_group_confusion
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

        x_range = [str(i) for i in range(15)]
        y_range = [str(i) for i in reversed(range(15))]

        fig = bop.figure(title=f'{self.split.capitalize()} Super Class Confusion',
                         x_range=x_range, y_range=y_range,
                         x_axis_location="above", width=356, height=712,
                         tools='hover,save', toolbar_location='below', tooltips=tooltips)
        fig.title.vertical_align = 'top'
        fig.grid.grid_line_color = None
        fig.axis.axis_line_color = None
        fig.axis.major_tick_line_color = None
        fig.axis.major_label_text_font_size = "8pt"
        fig.axis.major_label_standoff = 0
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
        elif self.split == 'dev':
            self.dev_group_confusion_fig = fig
        else:
            self.valid_group_confusion_fig = fig

        return self

    def __render_results_page(self):
        """Render batch and confusions matrices in the DOM. Add a periodic callback to play videos."""
        self.toggle = bom.Toggle(label='Rendering', active=True, button_type='success')
        self.toggle.on_click(self._on_toggle_click)

        self.doc.add_root(bol.widgetbox(self.toggle))

        self.doc.add_root(bol.layout(self.batch, sizing_mode='fixed'))
        self.doc.add_periodic_callback(self.__tick, 84)

    def __render_confusion_page(self):
        self.doc.add_root(
            bol.layout([[self.train_label_confusion_fig, self.train_group_confusion_fig]], sizing_mode='fixed'))
        self.doc.add_root(
            bol.layout([[self.dev_label_confusion_fig, self.dev_group_confusion_fig]], sizing_mode='fixed'))
        self.doc.add_root(
            bol.layout([[self.valid_label_confusion_fig, self.valid_group_confusion_fig]], sizing_mode='fixed'))

    def _on_toggle_click(self, _):
        if self.toggle.active:
            self.toggle.label = 'Rendering'
            self.toggle.button_type = 'success'
        else:
            self.toggle.label = 'Paused'
            self.toggle.button_type = 'danger'

    def __tick(self) -> None:
        """Increment step and update data source."""
        if self.toggle.active:
            self.step += 1
            self.source.data = {video.meta.id: [video.data[self.step % video.meta.length]] for video in self.videos}

    def __create_table_source(self, data: Dict[str, Any], name: str = '') -> (bom.ColumnDataSource, [bom.TableColumn]):
        """Create a table with 2 columns: Option, Value."""
        source = dict(opt=[], val=[])
        cols = [bom.TableColumn(field='opt', title=f'{name} Option'), bom.TableColumn(field='val', title='Value')]
        for opt, val in data.items():
            if val is None:
                continue
            if isinstance(val, dict):
                continue
            if isinstance(val, str) and val.startswith('<class'):
                val = val.replace("'>", "").split('.')[-1]
            if isinstance(val, list):
                val = ','.join(str(v) for v in val)
            source['opt'].append(opt)
            source['val'].append(val)
        if name == 'Run Options':
            source['opt'].append('data setting')
            source['val'].append(ct.SETTING)
        source = bom.ColumnDataSource(source)

        return source, cols

    def __create_line_plot(self, title: str, tip: str, x: str, ys: List[Tuple[str, str]],
                           fkwargs: Dict[str, any], lkwargs: Dict[str, any]) -> bom.Plot:
        """Create a line plot with hover tool from a stats DataFrame."""
        hover = bom.HoverTool(tooltips=[
            ('Epoch', '@epoch{0,}'),
            (f'{tip}', '@$name{0.0000}'),
        ], mode='vline')
        plot = bop.figure(title=title, tools=[hover, 'box_zoom', 'wheel_zoom', 'pan', 'reset'], **fkwargs)
        for i, (y, leg) in enumerate(ys):
            # noinspection PyUnresolvedReferences
            plot.line(x=x, y=y, source=self.stats, name=y, color=pal.Spectral6[i % len(ys)], legend=leg, **lkwargs)

        return plot
