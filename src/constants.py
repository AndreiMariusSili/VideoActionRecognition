import pathlib as pl
import os

ROOT = '..'
SETTING = 'dummy'

SMTH_WEBM_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'webm'))
SMTH_JPEG_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'jpeg'))
SMTH_META_DATA_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta'))
SMTH_META_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.merged.json'))
SMTH_META_TRAIN = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.train.json'))
SMTH_META_VALID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.valid.json'))
SMTH_META_TEST = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.test.json'))
SMTH_LABELS2ID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'template2id.json'))
SMTH_LABELS2GROUP = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'template2group.json'))
SMTH_GROUP_ID2GROUP_LABEL = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'id2group.json'))
SMTH_STATS_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'stats.merged.json'))

SMTH_TRAIN_DUMMY_SAMPLE = (100, 150)
SMTH_VALID_DUMMY_SAMPLE = (10, 15)

IMAGE_NET_STD_HEIGHT = 224
IMAGE_NET_STD_WIDTH = 224
IMAGE_NET_MEANS = (0.485, 0.456, 0.406)
IMAGE_NET_STDS = (0.229, 0.224, 0.225)

RUN_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'runs'))

I3D_TF_RGB_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'tf_rgb_checkpoint', 'model.ckpt'))
I3D_PT_RGB_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'pt_rgb_checkpoint', 'model.ckpt'))
I3D_TF_FLOW_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'tf_flow_checkpoint', 'model.ckpt'))
I3D_PT_FLOW_CHECKPOINT = pl.Path(
    os.path.join(ROOT, 'src', 'models', 'i3d', 'checkpoints', 'pt_flow_checkpoint', 'model.ckpt'))
I3D_PREPARE_DATASET = pl.Path(os.path.join(ROOT, 'data', 'i3d_prepare_dataset'))

STYLES = pl.Path(os.path.join(ROOT, 'src', 'assets', 'styles.css'))
