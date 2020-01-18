import databunch.video_meta as pim


class Label(object):
    data: int
    meta: pim.VideoMeta

    def __init__(self, meta: pim.VideoMeta):
        """Initialize a Label object from a row in the meta DataFrame and an integer identifier."""
        self.meta = meta
        self.data = meta.lid

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'({self.data} {self.meta.label})'

    def __repr__(self):
        return self.__str__()
