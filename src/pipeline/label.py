import bokeh.plotting as bop
import bokeh.models as bom

import pipeline as pipe


class Label(object):
    data: int
    meta: pipe.VideoMeta

    def __init__(self, meta: pipe.VideoMeta):
        """Initialize a Label object from a row in the meta DataFrame and an integer identifier."""
        self.meta = meta
        self.data = meta.template_id

    def show(self, fig: bop.Figure) -> None:
        """Compile to a Bokeh title object."""
        label = bom.Title(text=self.meta.template, text_font_size='8pt', align='left')
        fig.add_layout(label, 'above')

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'({self.data} {self.meta.template})'

    def __repr__(self):
        return self.__str__()
