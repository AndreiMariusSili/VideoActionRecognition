import os
import subprocess

import constants as ct
import helpers as hp
from env import logging


def _make_data_dirs():
    """Create directories.Throw error if already exist since this step must be done only once."""
    os.makedirs(hp.change_setting(ct.SMTH_META_DIR, 'full', 'dummy').as_posix())
    os.makedirs(hp.change_setting(ct.SMTH_WEBM_DIR, 'full', 'dummy').as_posix())
    os.makedirs(hp.change_setting(ct.SMTH_JPEG_DIR, 'full', 'dummy').as_posix())

    os.makedirs(hp.change_setting(ct.SMTH_META_DIR, 'dummy', 'full').as_posix())
    os.makedirs(hp.change_setting(ct.SMTH_WEBM_DIR, 'dummy', 'full').as_posix())
    os.makedirs(hp.change_setting(ct.SMTH_JPEG_DIR, 'dummy', 'full').as_posix())


def _move_meta() -> None:
    """Rename meta data to convention."""

    labels = hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full') / 'something-something-v2-labels.json'
    train = hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full') / 'something-something-v2-train.json'
    valid = hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full') / 'something-something-v2-validation.json'
    test = hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full') / 'something-something-v2-test.json'
    gid2labels = hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full') / 'gid2labels.json'
    labels2gid = hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full') / 'label2gid.json'

    labels.replace(hp.change_setting(ct.SMTH_LABEL2LID, 'dummy', 'full'))
    train.replace(hp.change_setting(ct.SMTH_META_TRAIN, 'dummy', 'full'))
    valid.replace(hp.change_setting(ct.SMTH_META_VALID, 'dummy', 'full'))
    test.replace(hp.change_setting(ct.SMTH_META_TEST, 'dummy', 'full'))
    gid2labels.replace(hp.change_setting(ct.SMTH_GID2LABELS, 'dummy', 'full'))
    labels2gid.replace(hp.change_setting(ct.SMTH_LABEL2GID, 'dummy', 'full'))


def _tar():
    """Un-tar the dataset and remove archive."""
    subprocess.call(f'cat 20bn-something-something-v2-?? | tar zx',
                    stdout=subprocess.PIPE,
                    cwd=hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full').as_posix(),
                    shell=True)


def _move_webm():
    """Move webm files to the webm folder."""
    webms = hp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full') / '20bn-something-something-v2'
    for webm in webms.glob('*.webm'):
        file_name = f'{webm.stem}{webm.suffix}'
        webm.replace(hp.change_setting(ct.SMTH_WEBM_DIR, 'dummy', 'full') / file_name)

    webms.rmdir()


def main():
    logging.info('Creating directories...')
    _make_data_dirs()
    logging.info('Moving meta files...')
    _move_meta()
    logging.info('Untaring...')
    _tar()
    logging.info('Moving webm files...')
    _move_webm()
    logging.info('Done.')
