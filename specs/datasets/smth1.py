import constants as ct
import options.data_options as do
import specs.datasets.common as dc

########################################################################################################################
# TRAIN DATASET
########################################################################################################################
train_dso = do.DataSetOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.SMTH_META_TRAIN_1,
    setting='train',
    use_flow=False
)
train_dso_flow = do.DataSetOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.HMDB_READ_JPEG,
    meta_path=ct.SMTH_META_TRAIN_1,
    setting='train',
    use_flow=True
)
########################################################################################################################
# DEV DATA
########################################################################################################################
dev_dso = do.DataSetOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.SMTH_READ_JPEG,
    meta_path=ct.SMTH_META_DEV_1,
    setting='eval',
    use_flow=False
)
dev_dso_flow = do.DataSetOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.SMTH_READ_JPEG,
    meta_path=ct.SMTH_META_DEV_1,
    setting='eval',
    use_flow=True
)
########################################################################################################################
# TEST DATA
########################################################################################################################
test_dso = do.DataSetOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.SMTH_READ_JPEG,
    meta_path=ct.SMTH_META_VALID_1,  # valid set is used as tests set since test set does not have ground truths.
    setting='eval',
    use_flow=False
)
test_dso_flow = do.DataSetOptions(
    root_path=ct.SMTH_ROOT_DIR,
    read_jpeg=ct.SMTH_READ_JPEG,
    meta_path=ct.SMTH_META_VALID_1,  # valid set is used as tests set since test set does not have ground truths.
    setting='eval',
    use_flow=True
)
########################################################################################################################
# DATA BUNCH
########################################################################################################################
dbo_4 = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.SMTH_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_4,
    train_dso=train_dso,
    dev_dso=dev_dso,
    test_dso=test_dso,
)
dbo_4_flow = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.SMTH_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_4,
    train_dso=train_dso_flow,
    dev_dso=dev_dso_flow,
    test_dso=test_dso_flow,
)
dbo_8 = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.SMTH_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_8,
    train_dso=train_dso,
    dev_dso=dev_dso,
    test_dso=test_dso,
)
dbo_8_flow = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.SMTH_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_8,
    train_dso=train_dso_flow,
    dev_dso=dev_dso_flow,
    test_dso=test_dso_flow,
)
dbo_16 = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.SMTH_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_16,
    train_dso=train_dso,
    dev_dso=dev_dso,
    test_dso=test_dso,
)
dbo_16_flow = do.DataBunch(
    shape='volume',
    cut=None,
    frame_size=224,
    stats_path=ct.SMTH_STATS_MERGED_1,
    distributed=False,
    dlo=dc.dlo,
    so=dc.so_16,
    train_dso=train_dso_flow,
    dev_dso=dev_dso_flow,
    test_dso=test_dso_flow,
)
