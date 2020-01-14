import os
import subprocess

import pandas as pd

import constants as ct
from env import logging

META_ARCHIVE = 'test_train_splits.rar'
VIDEO_ARCHIVE = 'hmdb51_org.rar'


def _make_data_dirs():
    """Create directories."""
    os.makedirs(ct.HMDB_META_DIR_1, exist_ok=True)
    os.makedirs(ct.HMDB_META_DIR_2, exist_ok=True)
    os.makedirs(ct.HMDB_META_DIR_3, exist_ok=True)
    os.makedirs(ct.HMDB_AVI_DIR, exist_ok=True)
    os.makedirs(ct.HMDB_JPEG_DIR, exist_ok=True)
    os.makedirs(ct.HMDB_TXT_DIR, exist_ok=True)
    os.makedirs(ct.HMDB_ARCHIVE_DIR, exist_ok=True)


def _extract():
    """Extract the dataset."""

    subprocess.call(f'7z x {(ct.HMDB_ROOT_DIR / VIDEO_ARCHIVE).as_posix()}',
                    stdout=subprocess.PIPE,
                    cwd=ct.HMDB_AVI_DIR.as_posix(),
                    shell=True)
    for archive in ct.HMDB_AVI_DIR.glob('*.rar'):
        subprocess.call(f'7z x {archive}',
                        stdout=subprocess.PIPE,
                        cwd=ct.HMDB_AVI_DIR.as_posix(),
                        shell=True)
        archive.unlink()
    subprocess.call(f'7z e {(ct.HMDB_ROOT_DIR / META_ARCHIVE).as_posix()}',
                    stdout=subprocess.PIPE,
                    cwd=ct.HMDB_TXT_DIR.as_posix(),
                    shell=True)


def _create_meta(split: str, verbose: bool = False):
    """Parse .txt files into meta.{split}.json"""
    a_split = ct.HMDB_TXT_DIR.glob(f'*_split{split}.txt')

    meta_train = []
    meta_test = []

    for path in a_split:
        label = path.stem.strip().split(f'_test_split{split}').pop(0)
        train, test, other = 0, 0, 0
        with open(path.as_posix(), 'r') as txt:
            for line in txt:
                name, _id = line.strip().split()
                stem = name.split('.avi').pop(0)
                video_path = (ct.HMDB_AVI_DIR / label / name).relative_to(ct.HMDB_ROOT_DIR).as_posix()
                image_folder_path = (ct.HMDB_JPEG_DIR / label / stem).relative_to(ct.HMDB_ROOT_DIR).as_posix()

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

    train_path = getattr(ct, f'HMDB_META_TRAIN_{split}')
    test_path = getattr(ct, f'HMDB_META_TEST_{split}')

    meta_train.set_index('id', drop=False, verify_integrity=True).to_json(train_path, orient='index')
    meta_test.set_index('id', drop=False, verify_integrity=True).to_json(test_path, orient='index')


def __create_labels(meta_train: pd.DataFrame) -> pd.DataFrame:
    """Create labels mapping from train meta.."""
    labels = pd.DataFrame(meta_train['label'].unique(), columns=['label']).reset_index().set_index('label')
    labels.columns = ['lid']

    return labels


def _cleanup():
    """Remove temporary directories and move archives to folder."""
    for path in ct.HMDB_TXT_DIR.glob("*"):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    ct.HMDB_TXT_DIR.rmdir()

    (ct.HMDB_ROOT_DIR / VIDEO_ARCHIVE).replace(ct.HMDB_ARCHIVE_DIR / VIDEO_ARCHIVE)
    (ct.HMDB_ROOT_DIR / META_ARCHIVE).replace(ct.HMDB_ARCHIVE_DIR / META_ARCHIVE)


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
