import numpy as np
import skvideo.io

from pipeline import _video_meta


class Video(object):
    data: np.ndarray
    meta: _video_meta.VideoMeta
    cut: int

    def __init__(self, meta: _video_meta.VideoMeta, cut: float):
        """Initialize a Video object from a row in the meta DataFrame."""
        assert 0.0 <= cut <= 1.0, f'Cut should be a value between 0.0 and 1.0. Received: {cut}.'

        self.meta = meta
        self.cut = int(self.meta.length * cut)
        self.data = skvideo.io.vread(meta.path, num_frames=self.cut)

    def show(self):
        """Compile to a Bokeh animation object."""
        pass

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'{self.meta.id} ({"x".join(map(str, self.data.shape))})'

    def __repr__(self):
        return self.__str__()

# class Video(core.ItemBase):
#     obj: List[torch.Tensor]
#     data: List[Any]
#     id: str
#
#     def __init__(self, frame_seq: List[vs.Image], _id: str):
#         super().__init__(frame_seq)
#
#         self.id = _id
#         self.obj = frame_seq
#         self.data = torch.stack([frame.data for frame in frame_seq], dim=0)
#
#     def to_gif(self):
#         frames = [vs.image2np(frame.data * 255).astype(core.np.uint8) for frame in self.obj]
#
#         return frames
#
#     def to_one(self):
#         return vs.image2np(self.obj[0].data * 255).astype(core.np.uint8)
#
#     def show(self, ax: core.plt.Axes = None, figsize: tuple = (3, 3), title: Optional[str] = None,
#              hide_axis: bool = True, **kwargs) -> List[AxesImage]:
#         """Show image on `ax` with `title`."""
#         if hide_axis:
#             ax.set_xticks([])
#             ax.set_yticks([])
#         ax.imshow(self.to_one())
#
#     def __str__(self):
#         return f'{self.id} ({len(self.obj)},{",".join(map(str, self.obj[0].shape))})'
#
#
# class VideoList(vs.ImageItemList):
#     cut: int
#     meta_path: core.Path
#     meta: core.pd.DataFrame
#
#     def __init__(self, items: List[core.PathOrStr], cut: int, meta: core.PathOrStr, **kwargs):
#         self.cut = cut
#         self.meta_path = core.Path(meta)
#         self.meta = hp.read_smth_meta(self.meta_path)
#         items = [core.Path(item) for item in items]
#
#         super().__init__(items, **kwargs)
#
#     def get(self, i) -> 'Video':
#         video: core.Path = self.items[i]
#         _id = video.name
#         frame_seq = self.open(video)
#
#         return Video(frame_seq, _id)
#
#     def open(self, folder: core.PathOrStr):
#         folder = core.Path(folder)
#         filenames = sorted(folder.glob('*'))
#
#         return [super(VideoList, self).open(filename) for filename in filenames]
#
#     def new(self, items, **kwargs):
#         return super().new(items, cut=self.cut, meta=self.meta_path, **kwargs)
#
#     def reconstruct(self, t: torch.Tensor, x: torch.Tensor = None) -> 'Video':
#         frames = t.shape[0]
#
#         return Video([super(VideoList, self).reconstruct(t[frame]) for frame in range(frames)], "0")
#
#     def show_xys(self, xs, ys, figsize: Tuple[int, int] = (12, 6), **kwargs):
#         """Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."""
#         rows = int(math.sqrt(len(xs)))
#         fig, axs = core.plt.subplots(rows, rows, figsize=figsize)
#
#         for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
#             xs[i].show(ax=ax, **kwargs)
#             ys[i].show(ax=ax, **kwargs)
#         core.plt.tight_layout()
#
#     def show_xyzs(self, xs, ys, zs, figsize: Tuple[int, int] = (12, 6), **kwargs):
#         """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
#         `kwargs` are passed to the show method."""
#         rows = int(math.sqrt(len(xs)))
#         fig, axs = core.plt.subplots(rows, rows, figsize=figsize)
#         fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
#         for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
#             xs[i].show(ax=ax, **kwargs)
#             ys[i].show(ax=ax, **kwargs)
#             zs[i].show(ax=ax, **kwargs)
#         core.plt.tight_layout()
#
#     @classmethod
#     def from_folder(cls, _input: core.PathOrStr = '.', extensions: Collection[str] = None, cut: int = 1,
#                     meta: core.PathOrStr = None, **kwargs) -> 'VideoList':
#         if meta is None:
#             raise ValueError('Must specify a list of path to the merged DataFrame.')
#         _input = core.Path(_input)
#         meta = core.Path(meta)
#         folders = list(sorted(_input.glob('[!.]*/[!.]*/[0-9]*')))
#
#         return cls(folders, cut, meta, path=_input)
#
#
# def get_max(videos: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
#     max_height, max_width = 0, 0
#     for video in videos:
#         _, _, height, width = video.shape
#         if height > max_height:
#             max_height = height
#         if width > max_width:
#             max_width = width
#
#     return max_height, max_width
#
#
# def get_padding_values(height: torch.Tensor, width: torch.Tensor,
#                        max_height: torch.Tensor, max_width: torch.Tensor) -> Tuple[torch.Tensor, ...]:
#     vertical = max_height - height
#     horizontal = max_width - width
#     if vertical % 2 == 0:
#         top = bottom = vertical // 2
#     else:
#         top = vertical // 2
#         bottom = (vertical // 2) + 1
#
#     if horizontal % 2 == 0:
#         left = right = horizontal // 2
#     else:
#         left = horizontal // 2
#         right = (horizontal // 2) + 1
#
#     return top, bottom, left, right
#
#
# def zero_pad(video: torch.Tensor, max_height: torch.Tensor, max_width: torch.Tensor) -> torch.Tensor:
#     _, _, height, width = video.shape
#
#     top, bottom, left, right = get_padding_values(height, width, max_height, max_width)
#
#     return torch.nn.ZeroPad2d((left, right, top, bottom))(video)
#
#
# def collate(batch: List[Tuple[Video, core.Category]]):
#     videos, targets = zip(*batch)
#
#     videos = list(map(lambda video: video.data, videos))
#     max_height, max_width = get_max(videos)
#     videos = list(map(zero_pad, videos, [max_height] * len(videos), [max_width] * len(videos)))
#     videos = torch.nn.utils.rnn.pad_sequence(videos, batch_first=True)
#
#     targets = torch_core.tensor(torch_core.to_data(targets))
#
#     return videos, targets
