import subprocess

import constants as ct
import helpers as hp
from env import logging


def _make_data_dirs():
    """Create directories."""
    os.makedirs(hp.change_setting(ct.SMTH_META_DIR, 'full', 'dummy').as_posix(), exist_ok=True)
    os.makedirs(hp.change_setting(ct.SMTH_WEBM_DIR, 'full', 'dummy').as_posix(), exist_ok=True)
    os.makedirs(hp.change_setting(ct.SMTH_JPEG_DIR, 'full', 'dummy').as_posix(), exist_ok=True)

    os.makedirs(hp.change_setting(ct.SMTH_META_DIR, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(hp.change_setting(ct.SMTH_WEBM_DIR, 'dummy', 'full').as_posix(), exist_ok=True)
    os.makedirs(hp.change_setting(ct.SMTH_JPEG_DIR, 'dummy', 'full').as_posix(), exist_ok=True)


def _extract():
    """Extract the dataset."""
    subprocess.call(f'tar -zxf webm.tar ',
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
    logging.info('Creating data directories...')
    _make_data_dirs()
    logging.info('Extracting webm files...')
    _extract()
    logging.info('Moving webm files...')
    _move_webm()
    logging.info('Done.')
