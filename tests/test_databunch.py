import pytest
import tqdm

import databunch.databunch as db
import specs

HMDB_DBO = [
    specs.datasets.hmdb1.dbo_4,
    specs.datasets.hmdb1.dbo_8,
    # specs.datasets.hmdb1.dbo_16,
    specs.datasets.hmdb1.dbo_4_flow,
    specs.datasets.hmdb1.dbo_8_flow,
    # specs.datasets.hmdb1.dbo_16_flow,
]
SMTH_DBO = [
    specs.datasets.smth1.dbo_4,
    specs.datasets.smth1.dbo_8,
    # specs.datasets.smth1.dbo_16
]


@pytest.mark.parametrize('dbo', HMDB_DBO + SMTH_DBO)
def test_databunch(dbo):
    dbo.dlo.timeout = 0
    dbo.dlo.num_workers = 6
    dbo.dlo.batch_size = 64
    dbo.dlo.shuffle = False
    dbo.cut = 1.0

    bunch = db.VideoDataBunch(dbo)
    tqdm.tqdm.write(str(bunch))

    for split in ['train', 'dev', 'test']:

        if split == 'train':
            dataset, dataloader = bunch.train_set, bunch.train_loader
        elif split == 'dev':
            dataset, dataloader = bunch.dev_set, bunch.dev_loader
        else:
            dataset, dataloader = bunch.test_set, bunch.test_loader

        pbar = tqdm.tqdm(total=len(dataset))
        for i, (x, y, *_) in enumerate(dataloader):
            pbar.update(x.shape[0])
        pbar.clear()
        pbar.close()

    tqdm.tqdm.write('')
