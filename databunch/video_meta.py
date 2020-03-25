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
