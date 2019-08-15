import os
import shutil
from typing import Tuple

import pandas as pd

import constants as ct
import helpers as hp
from env import logging

RANDOM_STATE = -1


def _create_dummy_groups() -> pd.DataFrame:
    """Sample a number of contrastive groups. Store as both label2gid and gid2labels."""
    global RANDOM_STATE
    RANDOM_STATE += 1
    logging.info(f'Current random seed: {RANDOM_STATE}')
    full_gid2labels = hp.read_smth_gid2labels(hp.change_setting(ct.SMTH_GID2LABELS, ct.SETTING, 'full').as_posix())
    group_counts = full_gid2labels.groupby('id', as_index=False).count()
    groups_of_2 = group_counts[group_counts['template'] == 2]
    sampled_groups = groups_of_2[['id']].sample(ct.SMTH_NUM_CONTRASTIVE_GROUPS_SAMPLES, random_state=RANDOM_STATE)
    mask = full_gid2labels['id'].isin(sampled_groups['id'])

    sample = full_gid2labels[mask].sort_values(by='id', ascending=True)

    sample['id'] = [i for i in range(ct.SMTH_NUM_CONTRASTIVE_GROUPS_SAMPLES) for _ in range(2)]
    _range = [i % 2 for i in range(ct.SMTH_NUM_CONTRASTIVE_GROUPS_SAMPLES * 2)]
    dummy_gid2labels = sample.copy(deep=True).set_index(['id', _range], drop=False, verify_integrity=True)
    dummy_gid2labels.index.map(lambda _tuple: (int(_tuple[0]), int(_tuple[1])))
    dummy_gid2labels.to_json(ct.SMTH_GID2LABELS, orient='index')

    dummy_label2gid = sample.copy(deep=True).set_index('template', drop=False, verify_integrity=True)
    dummy_label2gid.index.map(str)
    dummy_label2gid.to_json(ct.SMTH_LABEL2GID, orient='index')

    return sample


def _create_dummy_labels(full_labels: pd.DataFrame, dummy_labels2lid: pd.DataFrame, dummy_split: str) -> pd.DataFrame:
    """Create the dummy labels DataFrame by matching with the dummy labels2id templates and sampling from ."""
    mask = full_labels['template'].isin(dummy_labels2lid['template'])
    dummy_labels = full_labels[mask]

    dummy_labels.reset_index(drop=True).to_json(dummy_split, orient='records')

    return dummy_labels


def _create_dummy_labels2lid_lid2labels(contrastive_groups: pd.DataFrame) -> None:
    """Create labels2lid from contrastive groups."""
    full_labels2lid = hp.read_smth_label2lid(hp.change_setting(ct.SMTH_LABEL2LID, ct.SETTING, 'full').as_posix())
    dummy_labels2lid = contrastive_groups[['template']].join(full_labels2lid[['id']], on='template')
    dummy_labels2lid['id'] = range(0, len(dummy_labels2lid))

    dummy_labels2lid = dummy_labels2lid.set_index('template', drop=False, verify_integrity=True)
    dummy_labels2lid.to_json(ct.SMTH_LABEL2LID, orient='index')

    dummy_lid2labels = dummy_labels2lid.set_index('id', drop=False, verify_integrity=True)
    dummy_lid2labels.to_json(ct.SMTH_LID2LABEL, orient='index')


def _create_dummy_inputs(dummy_labels: pd.DataFrame) -> Tuple[str, str, int]:
    """Copy .webm files with an id in the dummy labels DataFrame to the dummy folder."""
    os.makedirs(ct.SMTH_WEBM_DIR, exist_ok=True)

    smth_full_raw_data = hp.change_setting(ct.SMTH_WEBM_DIR, ct.SETTING, 'full').as_posix()
    for _id in dummy_labels['id']:
        shutil.copy(f'{os.path.join(smth_full_raw_data, _id)}.webm', f'{os.path.join(ct.SMTH_WEBM_DIR, _id)}.webm')

    return smth_full_raw_data, ct.SMTH_WEBM_DIR, len(dummy_labels)


def main():
    """Create dummy sets for the something-something dataset based on a random selection of contrastive groups.
        The groups might be non-mutually-exclusive so the sampling is repeated until this condition is met."""
    try:
        logging.info('Creating something-something dummy dataset...')
        # create dummy labels
        contrastive_groups = None
        sample_has_duplicate_labels = True
        while sample_has_duplicate_labels:
            try:
                contrastive_groups = _create_dummy_groups()
                sample_has_duplicate_labels = False
            except ValueError:
                pass
        _create_dummy_labels2lid_lid2labels(contrastive_groups)
        # load dummy labels
        dummy_labels2lid = hp.read_smth_label2lid(ct.SMTH_LABEL2LID)
        logging.info('Loaded dummy labels2lid DataFrame.')

        splits = [
            (ct.SMTH_META_TRAIN, hp.change_setting(ct.SMTH_META_TRAIN, ct.SETTING, 'full'), 'train'),
            (ct.SMTH_META_VALID, hp.change_setting(ct.SMTH_META_VALID, ct.SETTING, 'full'), 'valid'),
        ]
        # for each split
        for dummy_split, full_split, split_name in splits:
            # load split
            full_labels = hp.read_smth_meta(full_split)
            logging.info(f'Loaded full labels DataFrame from {full_split}.')
            # create dummy label files
            dummy_labels = _create_dummy_labels(full_labels, dummy_labels2lid, dummy_split)
            logging.info(f'Created {ct.SETTING} {split_name} labels DataFrame with {len(dummy_labels)} entries.')
            # copy matching inputs to dummy location
            _from, _to, total = _create_dummy_inputs(dummy_labels)
            logging.info(f'Copied {total} inputs from {_from} to {_to}.')
        hp.notify('good', 'dummy_smth', 'Finished creating dummy set.')
        logging.info('...Done')
    except Exception:
        hp.notify('bad', 'dummy_smth', 'Error occurred while creating dummy set.')
