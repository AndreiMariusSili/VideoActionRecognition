from typing import List

import bokeh.layouts as bol
import bokeh.models as bom
import pandas as pd

import databunch.video_meta as pim


class Prediction(object):
    meta: pim.VideoResultMeta

    def __init__(self, meta: pim.VideoResultMeta):
        self.meta = meta

    def show(self, id2template: pd.DataFrame, css: List[str] = None) -> bol.Column:
        """Compile the predictions to a column of bokeh divs."""
        rows = []
        template = '<span style="font-size: 8px">{prediction}</span>'
        for pred, conf in [('pred_1', 'conf_1'), ('pred_2', 'conf_2')]:
            pred, conf = getattr(self.meta, f'top2_{pred}'), getattr(self.meta, f'top2_{conf}')
            pred_temp = id2template.loc[pred]['template']
            pred_div = bom.Div(text=template.format(prediction=pred_temp))
            conf_div = bom.Div(text=template.format(prediction=round(conf, 4)))
            if css is not None:
                pred_div.css_classes = css
                conf_div.css_classes = css
            pred_div = bol.widgetbox(pred_div, width=200)
            conf_div = bol.widgetbox(conf_div, width=50)
            if css is not None:
                pred_div.css_classes = css
                conf_div.css_classes = css

            rows.append(bol.row([pred_div, conf_div], height=20))
        return bol.column(rows)
