import os
import pathlib as pl

########################################################################################################################
# PROJECT DIRECTORIES
########################################################################################################################
SOURCE_ROOT = pl.Path(os.environ['MT_SOURCE'])
WORK_ROOT = pl.Path(os.environ['MT_WORK'])
RUNS_ROOT = (WORK_ROOT / 'runs').relative_to(WORK_ROOT)
DATA_ROOT = (WORK_ROOT / 'data').relative_to(WORK_ROOT)
########################################################################################################################
# GENERAL SETTINGS
########################################################################################################################
STYLES = SOURCE_ROOT / 'assets' / 'styles.css'
########################################################################################################################
# DATASET SETTINGS
#######################################################################################################################

########################################################################################################################
# SOMETHING-SOMETHING-V2 SETTINGS
#######################################################################################################################
SMTH_ROOT_DIR = DATA_ROOT / 'smth'
SMTH_META1_DIR = SMTH_ROOT_DIR / 'meta' / '1'
SMTH_WEBM_DIR = SMTH_ROOT_DIR / 'webm'
SMTH_JPEG_DIR = SMTH_ROOT_DIR / 'jpeg'
SMTH_ARCHIVE_DIR = SMTH_ROOT_DIR / 'archive'
SMTH_META_MERGED_1 = SMTH_META1_DIR / 'meta.merged.json'
SMTH_META_TRAIN_1 = SMTH_META1_DIR / 'meta.train.json'
SMTH_META_DEV_1 = SMTH_META1_DIR / 'meta.dev.json'
SMTH_META_VALID_1 = SMTH_META1_DIR / 'meta.valid.json'
SMTH_META_TEST_1 = SMTH_META1_DIR / 'meta.test.json'
SMTH_STATS_MERGED_1 = SMTH_META1_DIR / 'stats.merged.json'
SMTH_LID2GID_1 = SMTH_META1_DIR / 'lid2gid.json'

SMTH_LID2LABEL = SMTH_META1_DIR / 'lid2label.json'
SMTH_LABEL2LID = SMTH_META1_DIR / 'label2lid.json'
SMTH_GID2GROUP = SMTH_META1_DIR / 'gid2group.json'
SMTH_GROUP2GID = SMTH_META1_DIR / 'group2gid.json'
SMTH_GID2LABELS = SMTH_META1_DIR / 'gid2labels.json'
SMTH_LABEL2GID = SMTH_META1_DIR / 'label2gid.json'

SMTH_RUN_DIR_1 = RUNS_ROOT / 'smth1'

try:
    next((WORK_ROOT / SMTH_JPEG_DIR).glob('*/*/*.jpeg'))
    SMTH_READ_JPEG = True
except StopIteration:
    SMTH_READ_JPEG = False
########################################################################################################################
# HMDB 51 SETTINGS
########################################################################################################################
HMDB_ROOT_DIR = DATA_ROOT / 'hmdb'
HMDB_ARCHIVE_DIR = HMDB_ROOT_DIR / 'archive'
HMDB_TXT_DIR = HMDB_ROOT_DIR / 'txt'
HMDB_AVI_DIR = HMDB_ROOT_DIR / 'avi'
HMDB_JPEG_DIR = HMDB_ROOT_DIR / 'jpeg'
HMDB_META_DIR_1 = HMDB_ROOT_DIR / 'meta' / '1'
HMDB_META_MERGED_1 = HMDB_META_DIR_1 / 'meta.merged.json'
HMDB_META_TRAIN_1 = HMDB_META_DIR_1 / 'meta.train.json'
HMDB_META_DEV_1 = HMDB_META_DIR_1 / 'meta.dev.json'
HMDB_META_TEST_1 = HMDB_META_DIR_1 / 'meta.test.json'
HMDB_STATS_MERGED_1 = HMDB_META_DIR_1 / 'stats.merged.json'
HMDB_META_DIR_2 = HMDB_ROOT_DIR / 'meta' / '2'
HMDB_META_MERGED_2 = HMDB_META_DIR_2 / 'meta.merged.json'
HMDB_META_TRAIN_2 = HMDB_META_DIR_2 / 'meta.train.json'
HMDB_META_DEV_2 = HMDB_META_DIR_2 / 'meta.dev.json'
HMDB_META_TEST_2 = HMDB_META_DIR_2 / 'meta.test.json'
HMDB_STATS_MERGED_2 = HMDB_META_DIR_2 / 'stats.merged.json'
HMDB_META_DIR_3 = HMDB_ROOT_DIR / 'meta' / '3'
HMDB_META_MERGED_3 = HMDB_META_DIR_3 / 'meta.merged.json'
HMDB_META_TRAIN_3 = HMDB_META_DIR_3 / 'meta.train.json'
HMDB_META_DEV_3 = HMDB_META_DIR_3 / 'meta.dev.json'
HMDB_META_TEST_3 = HMDB_META_DIR_3 / 'meta.test.json'
HMDB_STATS_MERGED_3 = HMDB_META_DIR_3 / 'stats.merged.json'

HMDB_RUN_DIR_1 = RUNS_ROOT / 'hmdb1'
HMDB_RUN_DIR_2 = RUNS_ROOT / 'hmdb2'
HMDB_RUN_DIR_3 = RUNS_ROOT / 'hmdb3'

try:
    next((WORK_ROOT / HMDB_JPEG_DIR).glob('*/*/*.jpeg'))
    HMDB_READ_JPEG = True
except StopIteration:
    HMDB_READ_JPEG = False

########################################################################################################################
# MODEL SETTINGS
########################################################################################################################
RANDOM_STATE = 42
DEV_SIZE = 0.10
TSNE_TRAIN_SAMPLE_SIZE = 2048
VAE_NUM_SAMPLES_TRAIN = 1
VAE_NUM_SAMPLES_DEV = 4
VAE_NUM_SAMPLES_TEST = 4
