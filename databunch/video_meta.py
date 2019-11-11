import dataclasses


@dataclasses.dataclass
class VideoMeta:
    fields = ['id', 'framerate', 'length', 'height',
              'width', 'label', 'lid', 'video_path', 'image_path']
    id: int
    framerate: int
    length: int
    height: int
    width: int
    label: str
    lid: int
    video_path: str
    image_path: str


@dataclasses.dataclass
class VideoResultMeta:
    fields = ['id', 'framerate', 'length', 'height',
              'width', 'label', 'lid', 'video_path', 'image_path',
              'top1_pred', 'top1_conf',
              'top3_conf_1', 'top3_conf_2', 'top3_conf_3',
              'top3_pred_1', 'top3_pred_2', 'top3_pred_3']
    id: int
    framerate: int
    length: int
    height: int
    width: int
    label: str
    lid: int
    video_path: str
    image_path: str
    top1_pred: int
    top1_conf: float
    top3_conf_1: float
    top3_conf_2: float
    top3_conf_3: float
    top3_pred_1: int
    top3_pred_2: int
    top3_pred_3: int
