import json
import os
import subprocess

import pandas as pd

import constants as ct
import helpers as ghp
from env import logging

META_TRAIN_NAME = 'something-something-v2-train.json'
META_VALID_NAME = 'something-something-v2-validation.json'
META_TEST_NAME = 'something-something-v2-test.json'
LABELS_NAME = 'something-something-v2-labels.json'
GROUPS_NAME = 'contrastive_groups_list.txt'


def __video_path(row: pd.Series) -> str:
    """Create path to video relative to data root dir."""
    return (ct.WORK_ROOT / ct.SMTH_WEBM_DIR / row['template'] / f'{row.id}.webm') \
        .relative_to(ct.WORK_ROOT / ct.SMTH_ROOT_DIR) \
        .as_posix()


def __image_path(row: pd.Series) -> str:
    """Create path to image folder relative to data root dir."""
    return (ct.WORK_ROOT / ct.SMTH_JPEG_DIR / row['template'] / f'{row.id}') \
        .relative_to(ct.WORK_ROOT / ct.SMTH_ROOT_DIR) \
        .as_posix()


def _make_data_dirs():
    """Create directories."""
    os.makedirs(ct.WORK_ROOT / ct.SMTH_META1_DIR.as_posix(), exist_ok=True)
    os.makedirs(ct.WORK_ROOT / ct.SMTH_WEBM_DIR.as_posix(), exist_ok=True)
    os.makedirs(ct.WORK_ROOT / ct.SMTH_JPEG_DIR.as_posix(), exist_ok=True)
    os.makedirs(ct.WORK_ROOT / ct.SMTH_ARCHIVE_DIR.as_posix(), exist_ok=True)


def _create_meta():
    """Create meta files. Groups are read from .txt file and labels with no grp are assigned a new singular grp."""
    meta_train = pd.read_json((ct.WORK_ROOT / ct.SMTH_ROOT_DIR / META_TRAIN_NAME).as_posix(), orient='records')
    meta_valid = pd.read_json((ct.WORK_ROOT / ct.SMTH_ROOT_DIR / META_VALID_NAME).as_posix(), orient='records')
    meta_test = pd.read_json((ct.WORK_ROOT / ct.SMTH_ROOT_DIR / META_TEST_NAME).as_posix(), orient='records')
    items = [
        ['train', ct.WORK_ROOT / ct.SMTH_META_TRAIN_1, meta_train],
        ['validation', ct.WORK_ROOT / ct.SMTH_META_VALID_1, meta_valid],
        ['test', ct.WORK_ROOT / ct.SMTH_META_TEST_1, meta_test],
    ]

    labels = _create_labels()
    groups = _create_groups()
    lid2gid = _create_lid2gid(groups, labels)
    lid2gid.to_json(ct.WORK_ROOT / ct.SMTH_LID2GID_1, orient='index')

    for split, path, meta in items:
        if split in ['train', 'validation']:
            meta['template'] = meta['template'].str.replace(r'\[([ a-z A-Z]*)\]', r'\1')

            meta['video_path'] = meta.apply(__video_path, axis=1)
            meta['image_path'] = meta.apply(__image_path, axis=1)

            meta = meta.join(labels, on='template')
            meta = meta.drop(['label', 'placeholders'], axis=1)
            meta = meta.rename(columns={'template': 'label'})

        meta.to_json(path, orient='index')


def _create_lid2gid(groups: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Creates a one to many mapping from lid to gids."""
    lid2gid = labels.join(groups)
    max_gid = int(lid2gid['gid'].max())
    no_miss_groups = lid2gid['gid'].isnull().sum()
    lid2gid.loc[lid2gid['gid'].isnull(), 'gid'] = range(max_gid + 1, max_gid + 1 + no_miss_groups)
    lid2gid = lid2gid.astype({'gid': int, 'lid': int}).reset_index(drop=True)

    return lid2gid


def _create_labels():
    """Create the labels DataFrame from the raw json."""
    with open((ct.WORK_ROOT / ct.SMTH_ROOT_DIR / LABELS_NAME).as_posix(), 'r') as file:
        labels = json.load(file)
    return pd.DataFrame(labels.items(), columns=['label', 'lid']) \
        .astype({'label': str, 'lid': int}) \
        .set_index('label', verify_integrity=True)


def _create_groups() -> pd.DataFrame:
    """Create the groups DataFrame from the raw txt."""
    with open((ct.WORK_ROOT / ct.SMTH_ROOT_DIR / GROUPS_NAME).as_posix(), 'r') as file:
        txt = file.read()
    groups = [list(filter(lambda x: x and not x.startswith('#'), group.split('\n'))) for group in txt.split('\n\n')]

    data = []
    for gid, group in enumerate(groups):
        for label in group:
            data.append([label, gid])
    return pd.DataFrame(data, columns=['label', 'gid']) \
        .astype({'label': str, 'gid': int}) \
        .set_index('label')


def _extract():
    """Extract the dataset."""
    subprocess.call(f'cat 20bn-something-something-v2-?? | tar zx',
                    stdout=subprocess.PIPE,
                    cwd=ct.WORK_ROOT / ct.SMTH_ROOT_DIR.as_posix(),
                    shell=True)


def _move_webm():
    """Move webm files to the webm folder."""
    meta_train = ghp.read_meta(ct.WORK_ROOT / ct.SMTH_META_TRAIN_1).set_index('id', verify_integrity=True)
    meta_valid = ghp.read_meta(ct.WORK_ROOT / ct.SMTH_META_VALID_1).set_index('id', verify_integrity=True)
    meta = pd.concat([meta_train, meta_valid], verify_integrity=True)

    webms = ct.WORK_ROOT / ct.SMTH_ROOT_DIR / '20bn-something-something-v2'
    for webm in webms.glob('*.webm'):
        try:
            label = meta.loc[int(webm.stem), 'label']
        except KeyError:
            label = 'Unknown'
        file_name = f'{webm.stem}{webm.suffix}'
        file_path = ct.WORK_ROOT / ct.SMTH_WEBM_DIR / label / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        webm.replace(file_path)

    webms.rmdir()


def _cleanup():
    """Move all original files to an archive folder."""
    for path in (ct.WORK_ROOT / ct.SMTH_ROOT_DIR).glob('20bn-something-something-v2-??'):
        archive_path = ct.WORK_ROOT / ct.SMTH_ARCHIVE_DIR / path.name
        path.replace(archive_path.as_posix())

    for file in [META_TRAIN_NAME, META_VALID_NAME, META_TEST_NAME, LABELS_NAME, GROUPS_NAME]:
        root_path = ct.WORK_ROOT / ct.SMTH_ROOT_DIR / file
        archive_path = ct.WORK_ROOT / ct.SMTH_ARCHIVE_DIR / file
        root_path.replace(archive_path)


def main():
    logging.info('Creating data directories...')
    _make_data_dirs()
    logging.info('Creating meta files...')
    _create_meta()
    logging.info('Extracting webm files...')
    _extract()
    logging.info('Moving webm files...')
    _move_webm()
    logging.info('Cleaning up...')
    _cleanup()
    logging.info('Done.')
