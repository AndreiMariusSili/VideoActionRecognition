import pathlib as pl
import os


ROOT = '..'
SETTING = 'dummy'

SMTH_VIDEO_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'raw'))
SMTH_META_DATA_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta'))
SMTH_META_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.merged.json'))
SMTH_META_TRAIN = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.train.json'))
SMTH_META_VALID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.valid.json'))
SMTH_META_TEST = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.test.json'))
SMTH_LABELS2ID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'template2id.json'))
SMTH_STATS_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'stats.merged.json'))
SMTH_TRAIN_DUMMY_SAMPLE = (100, 150)
SMTH_VALID_DUMMY_SAMPLE = (10, 15)
