import pathlib as pt
import shutil
import typing as t

import numpy as np
import pandas as pd

import constants as ct
import env
import helpers as hp
import options.job_options as jo

EXCLUDED_WEBM_DIR: pt.Path = ct.WORK_ROOT / ct.SMTH_ARCHIVE_DIR / 'excluded' / 'webm'
EXCLUDED_META1_DIR: pt.Path = ct.WORK_ROOT / ct.SMTH_ARCHIVE_DIR / 'excluded' / 'meta' / '1'
CLASS_CARD_RANK: t.List[t.Tuple[pt.Path, int]] = []


def prepare_excluded_dir() -> int:
    count = 0
    EXCLUDED_WEBM_DIR.mkdir(parents=True, exist_ok=True)
    for class_dir in EXCLUDED_WEBM_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        shutil.move(class_dir.as_posix(), ct.WORK_ROOT / ct.SMTH_WEBM_DIR.as_posix())
        count += 1

    EXCLUDED_META1_DIR.mkdir(parents=True, exist_ok=True)
    for meta in EXCLUDED_META1_DIR.iterdir():
        if meta.suffix != 'json':
            continue
        shutil.move(meta.as_posix(), ct.WORK_ROOT / ct.SMTH_META1_DIR.as_posix())

    return count


def create_class_card_rank():
    for class_dir in (ct.WORK_ROOT / ct.SMTH_WEBM_DIR).iterdir():
        if not class_dir.is_dir():
            continue
        card = len(list(class_dir.glob('*.webm')))
        CLASS_CARD_RANK.append((class_dir, card))

    CLASS_CARD_RANK.sort(key=lambda x: x[1], reverse=True)
    CLASS_CARD_RANK.append(CLASS_CARD_RANK.pop(0))  # most frequent class is "Unknown", which should be removed always.


def select_top_classes(num_classes: int):
    train_meta = hp.read_meta(ct.SMTH_META_TRAIN_1)
    valid_meta = hp.read_meta(ct.SMTH_META_VALID_1)

    for class_dir, class_card in CLASS_CARD_RANK[num_classes:]:
        shutil.move(class_dir.as_posix(), (EXCLUDED_WEBM_DIR / class_dir.name).as_posix())
        train_meta = train_meta[train_meta['label'] != class_dir.name]
        valid_meta = valid_meta[valid_meta['label'] != class_dir.name]
    lids = train_meta['lid'].unique()
    lids = pd.DataFrame(
        index=lids,
        data=np.arange(len(lids)),
        columns=['new_lid']
    )
    train_meta = train_meta.join(lids, on='lid')
    train_meta['lid'] = train_meta['new_lid']
    train_meta = train_meta.drop(labels='new_lid', axis=1)
    valid_meta = valid_meta.join(lids, on='lid')
    valid_meta['lid'] = valid_meta['new_lid']
    valid_meta = valid_meta.drop(labels='new_lid', axis=1)

    shutil.copy(str(ct.WORK_ROOT / ct.SMTH_META_TRAIN_1), str(EXCLUDED_META1_DIR / ct.SMTH_META_TRAIN_1.name))
    shutil.copy(str(ct.WORK_ROOT / ct.SMTH_META_VALID_1), str(EXCLUDED_META1_DIR / ct.SMTH_META_VALID_1.name))
    train_meta.to_json(ct.WORK_ROOT / ct.SMTH_META_TRAIN_1, orient='index')
    valid_meta.to_json(ct.WORK_ROOT / ct.SMTH_META_VALID_1, orient='index')


def main(opts: jo.SelectSubsetOptions):
    """Select ct.SMTH_SUBSET_CLASS_CARD number of num_classes to create a subset of the original dataset. All excluded
    num_classes are moved to EXCLUDED_WEBM_DIR. Selects top num_classes based on number of examples in a class.

    :return:
    """
    env.LOGGER.info(f'Selecting subset of {opts.num_classes} classes for smth dataset...')
    prepare_excluded_dir()
    create_class_card_rank()
    select_top_classes(opts.num_classes)
    env.LOGGER.info('Done.')
