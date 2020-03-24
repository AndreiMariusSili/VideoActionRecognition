from dataclasses import asdict

from torch.utils import data as thd

import databunch.dataset as dataset
import helpers as hp
import options.data_options as do


class VideoDataBunch(object):
    def __init__(self, db_opts: do.DataBunch):
        assert db_opts.shape in ['stack', 'volume'], f'Unknown shape {db_opts.shape}.'
        self.dbo = db_opts
        self.train_dso = self.dbo.train_dso
        self.dev_dso = self.dbo.dev_dso
        self.test_dso = self.dbo.test_dso
        self.stats = hp.read_stats(self.dbo.stats_path)

        self.train_set = dataset.VideoDataset(self.dbo.cut, self.dbo.frame_size, self.train_dso, self.dbo.so)
        self.dev_set = dataset.VideoDataset(self.dbo.cut, self.dbo.frame_size, self.dev_dso, self.dbo.so)
        self.test_set = dataset.VideoDataset(self.dbo.cut, self.dbo.frame_size, self.test_dso, self.dbo.so)

        self.lids = self.train_set.lids

        self.train_sampler = None
        self.dev_sampler = None
        self.test_sampler = None
        if self.dbo.distributed:
            self.dbo.dlo.shuffle = False
            self.train_sampler = thd.distributed.DistributedSampler(self.train_set)
            self.dev_sampler = thd.distributed.DistributedSampler(self.dev_set)
            self.test_sampler = thd.distributed.DistributedSampler(self.test_set)

        self.train_loader = thd.DataLoader(self.train_set,
                                           collate_fn=dataset.collate,
                                           sampler=self.train_sampler,
                                           worker_init_fn=dataset.init_worker,
                                           **asdict(self.dbo.dlo))
        self.dbo.dlo.shuffle = False
        self.dev_loader = thd.DataLoader(self.dev_set,
                                         collate_fn=dataset.collate,
                                         sampler=self.dev_sampler,
                                         worker_init_fn=dataset.init_worker,
                                         **asdict(self.dbo.dlo))
        self.test_loader = thd.DataLoader(self.test_set,
                                          collate_fn=dataset.collate,
                                          sampler=self.test_sampler,
                                          worker_init_fn=dataset.init_worker,
                                          **asdict(self.dbo.dlo))

    def __str__(self):
        return (f"""Something-Something-v2 DataBunch.
            [DataBunch config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dbo).items())}] 
            [Train Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.train_dso).items())}]
            [Dev Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dev_dso).items())}]
            [Test Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.test_dso).items())}]
            [DataLoader Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dbo.dlo).items())}]
            [Train Set: {self.train_set}]
            [Dev Set: {self.dev_set}]
            [Test Set: {self.test_set}]""")
