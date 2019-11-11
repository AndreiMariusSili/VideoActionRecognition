import bokeh.models as bom
import bokeh.plotting as bop

import databunch.video_meta as pim


class Label(object):
    data: int
    meta: pim.VideoMeta

    def __init__(self, meta: pim.VideoMeta):
        """Initialize a Label object from a row in the meta DataFrame and an integer identifier."""
        self.meta = meta
        self.data = meta.lid

    def show(self, fig: bop.Figure) -> None:
        """Compile to a Bokeh title object."""
        label = bom.Title(text=self.meta.label, text_font_size='8pt', align='left')
        fig.add_layout(label, 'above')

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'({self.data} {self.meta.label})'

    def __repr__(self):
        return self.__str__()
