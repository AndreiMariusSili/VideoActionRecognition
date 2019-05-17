import json
import os
import pathlib as pl

from torch import cuda

SLACK_NOTIFICATION_URL = 'https://hooks.slack.com/services/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'
COMPUTE_ENGINE_ICON = 'https://png2.kisspng.com/sh/6e14a20957f7cf13d844ea472ad8fce4/L0KzQYm3U8I5N5V4j5H0aYP2gLBuTfdwd5hxfZ9sbHB4dH73jPF1bpD3hZ9wb3BqfLa0gB9ueKZ5fZ9ubnfsfra0gBxwfZUye954dXSwc7F0kQV1cZ9sRadqYnO5RLO6UcI1a5Y3RqQEN0G8SIW3UcUzOmMATasDN0C6RXB3jvc=/kisspng-google-cloud-platform-google-compute-engine-cloud-cloud-computing-5abc64b3124ce2.297198401522295987075.png',

ROOT = '..'
SETTING = 'dummy'

SMTH_WEBM_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'webm'))
SMTH_JPEG_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'jpeg'))
SMTH_META_DATA_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta'))
SMTH_META_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.merged.json'))
SMTH_META_TRAIN = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.train.json'))
SMTH_META_VALID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.valid.json'))
SMTH_META_TEST = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.test.json'))
SMTH_STATS_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'stats.merged.json'))
SMTH_GID2LABELS = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'gid2labels.json'))
SMTH_LABEL2LID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'label2lid.json'))
SMTH_LID2LABEL = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'lid2label.json'))
SMTH_LABEL2GID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'label2gid.json'))
SMTH_GID2GROUP = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'gid2group.json'))

SMTH_RUN_DIR = pl.Path(os.path.join(ROOT, 'runs', SETTING, 'smth'))

SMTH_NUM_CONTRASTIVE_GROUPS_SAMPLES = 15
SMTH_TRAIN_DUMMY_SAMPLE = (2, 3)
SMTH_VALID_DUMMY_SAMPLE = (1, 2)

SMTH_NUM_CLASSES = None
if os.path.exists(SMTH_LABEL2LID):
    with open(SMTH_LABEL2LID, 'r') as file:
        SMTH_NUM_CLASSES = len(json.load(file))

IMAGE_NET_STD_HEIGHT = 224
IMAGE_NET_STD_WIDTH = 224
IMAGE_NET_MEANS = (0.485, 0.456, 0.406)
IMAGE_NET_STDS = (0.229, 0.224, 0.225)

I3D_TF_RGB_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'tf_rgb_checkpoint', 'model.ckpt'))
I3D_PT_RGB_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'pt_rgb_checkpoint', 'model.ckpt'))
I3D_TF_FLOW_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'tf_flow_checkpoint', 'model.ckpt'))
I3D_PT_FLOW_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'pt_flow_checkpoint', 'model.ckpt'))
I3D_PREPARE_DATASET = pl.Path(os.path.join(ROOT, 'data', 'i3d'))

STYLES = pl.Path(os.path.join(ROOT, 'src', 'assets', 'styles.css'))

NUM_DEVICES = cuda.device_count() if cuda.device_count() > 0 else 1

TSNE_SAMPLE_SIZE = 50
