import constants as ct
import options.data_options as do
import specs.datasets.common as dc

########################################################################################################################
# TRAIN DATASET
########################################################################################################################
train_dso = do.DataSetOptions(
    root_path=ct.HMDB_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.HMDB_META_TRAIN_1,
    setting='train',
)
########################################################################################################################
# DEV DATA
########################################################################################################################
dev_dso = do.DataSetOptions(
    root_path=ct.HMDB_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.HMDB_META_DEV_1,
    setting='eval',
)
########################################################################################################################
# TEST DATA
########################################################################################################################
test_dso = do.DataSetOptions(
    root_path=ct.HMDB_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.HMDB_META_TEST_1,
    setting='eval',
)
########################################################################################################################
# DATA BUNCH
########################################################################################################################
dbo_4 = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.HMDB_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_small,
    train_dso=train_dso,
    dev_dso=dev_dso,
    test_dso=test_dso,
)
dbo_16 = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.HMDB_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_large,
    train_dso=train_dso,
    dev_dso=dev_dso,
    test_dso=test_dso,
)
