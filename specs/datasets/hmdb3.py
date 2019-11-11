import constants as ct
import options.data_options as do
import specs.datasets.common as dc

########################################################################################################################
# TRAIN DATASET OPTIONS
########################################################################################################################
train_do = do.DataOptions(
    root_path=ct.HMDB_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.HMDB_META_TRAIN_3,
    setting='train',
)
train_dso = do.DataSetOptions(
    do=train_do,
    so=dc.so
)
########################################################################################################################
# DEV DATA
########################################################################################################################
dev_do = do.DataOptions(
    root_path=ct.HMDB_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.HMDB_META_DEV_3,
    setting='eval',
)
dev_dso = do.DataSetOptions(
    do=dev_do,
    so=dc.so
)
########################################################################################################################
# TEST DATA
########################################################################################################################
test_do = do.DataOptions(
    root_path=ct.HMDB_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.HMDB_META_TEST_3,
    setting='eval',
)
test_dso = do.DataSetOptions(
    do=test_do,
    so=dc.so
)
########################################################################################################################
# DATA BUNCH OPTIONS
########################################################################################################################
dbo = do.DataBunchOptions(
    shape='volume',
    cut=None,
    stats_path=ct.HMDB_STATS_MERGED_3,
    frame_size=224,
    distributed=False,
    dlo=dc.dlo,
    train_dso=train_dso,
    dev_dso=dev_dso,
    test_dso=test_dso,
)
