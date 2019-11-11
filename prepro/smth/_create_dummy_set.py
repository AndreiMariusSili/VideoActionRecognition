import shutil

import numpy as np
import pandas as pd

import constants as ct
import helpers as ghp
from env import logging


def _create_dummy_lid2gid() -> pd.DataFrame:
    """Sample a number of contrastive groups. Store as both label2gid and gid2labels."""
    random_state = -1
    dummy_lid2gid = None
    sample_has_duplicate_labels = True
    while sample_has_duplicate_labels:
        try:
            random_state += 1
            # Read full lid2gid and select only groups of cardinality 2.
            full_lid2gid = ghp.read_lid2gid(ghp.change_setting(ct.SMTH_LID2GID_1, 'dummy', 'full'))
            group_counts = full_lid2gid.groupby('gid', as_index=False).count()
            groups_of_2 = group_counts[group_counts['lid'] == 2]
            gid_sample = groups_of_2[['gid']].sample(ct.SMTH_NUM_CONTRASTIVE_GROUPS_SAMPLES, random_state=random_state)
            sample_mask = full_lid2gid['gid'].isin(gid_sample['gid'])
            dummy_lid2gid = full_lid2gid[sample_mask]

            # check if lids are unique.
            dummy_lid2gid.set_index('lid', drop=True, verify_integrity=True).reset_index()
            sample_has_duplicate_labels = False

            # remap lids and gids for dummy set
            dummy_lid = dummy_lid2gid.groupby('lid').count().drop('gid', axis=1)
            dummy_lid['dummy_lid'] = np.arange(0, len(dummy_lid.index))
            dummy_gid = dummy_lid2gid.groupby('gid').count().drop('lid', axis=1)
            dummy_gid['dummy_gid'] = np.arange(0, len(dummy_gid.index))
            dummy_lid2gid = dummy_lid2gid \
                .join(dummy_lid, on='lid', how='inner') \
                .join(dummy_gid, on='gid', how='inner') \
                .sort_index()
        except ValueError:
            pass

    return dummy_lid2gid


def _create_dummy_meta(full_meta: pd.DataFrame, dummy_lid2gid: pd.DataFrame):
    dummy_lids = dummy_lid2gid[['lid', 'dummy_lid']].set_index('lid', verify_integrity=True)
    return full_meta \
        .join(dummy_lids, on='lid', how='inner') \
        .drop('lid', axis=1) \
        .rename({'dummy_lid': 'lid'}, axis=1) \
        .sort_index()


def _create_dummy_labels(full_labels: pd.DataFrame, dummy_labels2lid: pd.DataFrame, dummy_split: str) -> pd.DataFrame:
    """Create the dummy labels DataFrame by matching with the dummy labels2id templates and sampling from ."""
    mask = full_labels['template'].isin(dummy_labels2lid['template'])
    dummy_labels = full_labels[mask]

    dummy_labels.reset_index(drop=True).to_json(dummy_split, orient='records')

    return dummy_labels


def _create_dummy_inputs(dummy_meta: pd.DataFrame) -> None:
    """Copy .webm form the dummy set into the corresponding dummy folder."""
    full_root_dir = ghp.change_setting(ct.SMTH_ROOT_DIR, 'dummy', 'full')
    dummy_root_dir = ghp.change_setting(ct.SMTH_ROOT_DIR, 'full', 'dummy')
    for video_path in dummy_meta['video_path']:
        src = full_root_dir / video_path
        dst = dummy_root_dir / video_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src.as_posix(), dst.as_posix())


def main(split: int):
    """Create dummy sets for the something-something dataset based on a random selection of contrastive groups.
        The groups might be non-mutually-exclusive so the sampling is repeated until this condition is met."""
    logging.info('Creating smth dummy lid2gid...')
    dummy_lid2gid = _create_dummy_lid2gid()
    dummy_lid2gid.to_json(ghp.change_setting(ct.SMTH_LID2GID_1, 'full', 'dummy'))

    splits = [
        (ghp.change_setting(ct.SMTH_META_TRAIN_1, 'full', 'dummy'),
         ghp.change_setting(ct.SMTH_META_TRAIN_1, 'dummy', 'full'),
         'train'),
        (ghp.change_setting(ct.SMTH_META_VALID_1, 'full', 'dummy'),
         ghp.change_setting(ct.SMTH_META_VALID_1, 'dummy', 'full'),
         'valid'),
    ]
    # for each split
    for dummy_meta_path, full_meta_path, split_name in splits:
        # load split
        logging.info(f'Creating smth dummy meta for {split_name}...')
        full_meta = ghp.read_meta(full_meta_path)
        dummy_meta = _create_dummy_meta(full_meta, dummy_lid2gid)
        dummy_meta.to_json(dummy_meta_path, orient='index')

        logging.info(f'Creating smth dummy inputs for {split_name}...')
        _create_dummy_inputs(dummy_meta)
    logging.info('...Done')
