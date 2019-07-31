import json
import os
import pathlib as pl

SLACK_NOTIFICATION_URL = 'https://hooks.slack.com/services/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'

ROOT = os.environ['MT_ROOT']
SETTING = 'dummy'

SMTH_WEBM_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'webm'))
SMTH_JPEG_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'jpeg'))
SMTH_META_DATA_DIR = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta'))
SMTH_META_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.merged.json'))
SMTH_META_TRAIN = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.train.json'))
SMTH_META_DEV = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.dev.json'))
SMTH_META_VALID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.valid.json'))
SMTH_META_TEST = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'meta.test.json'))
SMTH_STATS_MERGED = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'stats.merged.json'))
SMTH_GID2LABELS = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'gid2labels.json'))
SMTH_LABEL2LID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'label2lid.json'))
SMTH_LID2LABEL = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'lid2label.json'))
SMTH_LABEL2GID = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'label2gid.json'))
SMTH_GID2GROUP = pl.Path(os.path.join(ROOT, 'data', SETTING, 'smth', 'meta', 'gid2group.json'))

SMTH_NUM_CONTRASTIVE_GROUPS_SAMPLES = 15
DEV_SIZE = 0.20
READ_JPEG = SMTH_JPEG_DIR.exists()

SMTH_RUN_DIR = pl.Path(os.path.join(ROOT, 'runs', SETTING, 'smth'))

SMTH_NUM_CLASSES = None
if os.path.exists(SMTH_LABEL2LID):
    with open(SMTH_LABEL2LID, 'r') as file:
        SMTH_NUM_CLASSES = len(json.load(file))

IMAGE_NET_MEANS = (0.485, 0.456, 0.406)
IMAGE_NET_STDS = (0.229, 0.224, 0.225)

STYLES = pl.Path(os.path.join(ROOT, 'src', 'assets', 'styles.css'))

RANDOM_STATE = 0
VAE_NUM_SAMPLES_DEV = 10
VAE_NUM_SAMPLES_VALID = 50
TSNE_SAMPLE_SIZE = 200

KLD_STEP_INTERVAL = 5
KLD_STEP_SIZE = 0.10

EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5
