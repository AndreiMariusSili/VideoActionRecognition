import constants as ct
import options.data_options as po
import specs.datasets.common as dc

########################################################################################################################
# TRAIN DATASET OPTIONS
########################################################################################################################
train_do = po.DataOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.SMTH_READ_JPEG,
    meta_path=ct.SMTH_META_TRAIN_1,
    setting='train',
)
train_dso = po.DataSetOptions(
    do=train_do,
    so=dc.so
)
########################################################################################################################
# DEV DATA
########################################################################################################################
dev_do = po.DataOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.SMTH_READ_JPEG,
    meta_path=ct.SMTH_META_TRAIN_1,
    setting='eval',
)
dev_dso = po.DataSetOptions(
    do=dev_do,
    so=dc.so
)
########################################################################################################################
# TEST DATA
########################################################################################################################
test_do = po.DataOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.SMTH_READ_JPEG,
    meta_path=ct.SMTH_META_TRAIN_1,
    setting='eval',
)
test_dso = po.DataSetOptions(
    do=test_do,
    so=dc.so
)
########################################################################################################################
# DATA BUNCH OPTIONS
########################################################################################################################
dbo = po.DataBunchOptions(
    shape='volume',
    cut=1.0,
    stats_path=ct.SMTH_STATS_MERGED_1,
    frame_size=224,
    distributed=False,
    dlo=dc.dlo,
    train_dso=train_dso,
    dev_dso=dev_dso,
    test_dso=test_dso
)
