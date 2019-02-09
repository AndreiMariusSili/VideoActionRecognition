import dataclasses
from typing import List


@dataclasses.dataclass
class VideoMeta:
    fields = ['id', 'framerate', 'length', 'height', 'width', 'label', 'path', 'placeholders', 'template', 'template_id']
    id: int
    framerate: int
    length: int
    height: int
    width: int
    label: str
    path: str
    placeholders: List[str]
    template: str
    template_id: int
