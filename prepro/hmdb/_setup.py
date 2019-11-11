import os
import subprocess

import pandas as pd

import constants as ct
import helpers as ghp
from env import logging

META_ARCHIVE = 'test_train_splits.rar'
VIDEO_ARCHIVE = 'hmdb51_org.rar'

FULL_HMDB_ROOT_DIR = ghp.change_setting(ct.HMDB_ROOT_DIR, 'dummy', 'full')
FULL_HMDB_AVI_DIR = ghp.change_setting(ct.HMDB_AVI_DIR, 'dummy', 'full')
FULL_HMDB_TXT_DIR = ghp.change_setting(ct.HMDB_TXT_DIR, 'dummy', 'full')
FULL_JPEG_DIR = ghp.change_setting(ct.HMDB_JPEG_DIR, 'dummy', 'full')
FULL_ARCHIVE_DIR = ghp.change_setting(ct.HMDB_ARCHIVE_DIR, 'dummy', 'full')


def _make_data_dirs():
    """Create directories."""
    os.makedirs(ghp.change_setting(ct.HMDB_META_DIR_1, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_META_DIR_2, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_META_DIR_3, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_AVI_DIR, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_JPEG_DIR, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_TXT_DIR, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_ARCHIVE_DIR, 'dummy', 'full').as_posix(), exist_ok=True)

    os.makedirs(ghp.change_setting(ct.HMDB_META_DIR_1, 'full', 'dummy').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_META_DIR_2, 'full', 'dummy').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_META_DIR_3, 'full', 'dummy').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_AVI_DIR, 'full', 'dummy').as_posix(), exist_ok=True)
    os.makedirs(ghp.change_setting(ct.HMDB_JPEG_DIR, 'full', 'dummy').as_posix(), exist_ok=True)


def _extract():
    """Extract the dataset."""

    subprocess.call(f'7z x {(FULL_HMDB_ROOT_DIR / VIDEO_ARCHIVE).as_posix()}',
                    stdout=subprocess.PIPE,
                    cwd=FULL_HMDB_AVI_DIR.as_posix(),
                    shell=True)
    for archive in FULL_HMDB_AVI_DIR.glob('*.rar'):
        subprocess.call(f'7z x {archive}',
                        stdout=subprocess.PIPE,
                        cwd=FULL_HMDB_AVI_DIR.as_posix(),
                        shell=True)
        archive.unlink()
    subprocess.call(f'7z e {(FULL_HMDB_ROOT_DIR / META_ARCHIVE).as_posix()}',
                    stdout=subprocess.PIPE,
                    cwd=FULL_HMDB_TXT_DIR.as_posix(),
                    shell=True)


def _create_meta(split: str, verbose: bool = False):
    """Parse .txt files into meta.{split}.json"""
    a_split = FULL_HMDB_TXT_DIR.glob(f'*_split{split}.txt')

    meta_train = []
    meta_test = []

    for path in a_split:
        label = path.stem.strip().split(f'_test_split{split}').pop(0)
        train, test, other = 0, 0, 0
        with open(path.as_posix(), 'r') as txt:
            for line in txt:
                name, _id = line.strip().split()
                stem = name.split('.avi').pop(0)
                video_path = (FULL_HMDB_AVI_DIR / label / name).relative_to(FULL_HMDB_ROOT_DIR).as_posix()
                image_folder_path = (FULL_JPEG_DIR / label / stem).relative_to(FULL_HMDB_ROOT_DIR).as_posix()

                if _id == "1":
                    meta_train.append((name, video_path, image_folder_path, label))
                    train += 1
                elif _id == "2":
                    meta_test.append((name, video_path, image_folder_path, label))
                    test += 1
                else:
                    other += 1

        if verbose:
            msg = f'{path.as_posix():90s} | train: {train:2d} | test: {test:2d} | other: {other:2d}'
            logging.info(msg)

    meta_train = pd.DataFrame.from_records(meta_train, columns=['id', 'video_path', 'image_path', 'label'])
    meta_test = pd.DataFrame.from_records(meta_test, columns=['id', 'video_path', 'image_path', 'label'])

    labels = __create_labels(meta_train)

    meta_train = meta_train.join(labels, on='label', how='inner')
    meta_test = meta_test.join(labels, on='label', how='inner')

    train_path = ghp.change_setting(getattr(ct, f'HMDB_META_TRAIN_{split}'), 'dummy', 'full')
    test_path = ghp.change_setting(getattr(ct, f'HMDB_META_TEST_{split}'), 'dummy', 'full')

    meta_train.set_index('id', drop=False, verify_integrity=True).to_json(train_path, orient='index')
    meta_test.set_index('id', drop=False, verify_integrity=True).to_json(test_path, orient='index')


def __create_labels(meta_train: pd.DataFrame) -> pd.DataFrame:
    """Create labels mapping from train meta.."""
    labels = pd.DataFrame(meta_train['label'].unique(), columns=['label']).reset_index().set_index('label')
    labels.columns = ['lid']

    return labels


def _cleanup():
    """Remove temporary directories and move archives to folder."""
    for path in FULL_HMDB_TXT_DIR.glob("*"):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    FULL_HMDB_TXT_DIR.rmdir()

    (FULL_HMDB_ROOT_DIR / VIDEO_ARCHIVE).replace(FULL_ARCHIVE_DIR / VIDEO_ARCHIVE)
    (FULL_HMDB_ROOT_DIR / META_ARCHIVE).replace(FULL_ARCHIVE_DIR / META_ARCHIVE)


def main():
    logging.info('Creating data directories...')
    _make_data_dirs()
    logging.info('Extracting dataset files...')
    _extract()
    logging.info('Creating meta files...')
    for split in ['1', '2', '3']:
        _create_meta(split)
    logging.info('Cleaning up...')
    _cleanup()
    logging.info('Done.')
