import dataclasses
from typing import List


@dataclasses.dataclass
class VideoMeta:
    fields = ['id', 'framerate', 'length', 'height',
              'width', 'label', 'webm_path', 'jpeg_path',
              'placeholders', 'template', 'template_id']
    id: int
    framerate: int
    length: int
    height: int
    width: int
    label: str
    webm_path: str
    jpeg_path: str
    placeholders: List[str]
    template: str
    template_id: int


@dataclasses.dataclass
class VideoResultMeta:
    fields = ['id', 'framerate', 'length', 'height',
              'width', 'label', 'webm_path', 'jpeg_path',
              'placeholders', 'template', 'template_id',
              'top1_pred', 'top1_conf',
              'top3_conf_1', 'top3_conf_2', 'top3_conf_3',
              'top3_pred_1', 'top3_pred_2', 'top3_pred_3']
    id: int
    framerate: int
    length: int
    height: int
    width: int
    label: str
    webm_path: str
    jpeg_path: str
    placeholders: List[str]
    template: str
    template_id: int
    top1_pred: int
    top1_conf: float
    top3_conf_1: float
    top3_conf_2: float
    top3_conf_3: float
    top3_pred_1: int
    top3_pred_2: int
    top3_pred_3: int
