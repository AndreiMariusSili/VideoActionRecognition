import torchvision as tv
import tensorflow as tf
import torch as th

from models.i3d import _helpers as hp
from models.i3d import _i3dtf
from models.i3d import _i3dpt
import constants as ct


def prepare(tf_checkpoint: str, pt_checkpoint: str, batch_size: int, modality: str):
    if modality not in ['rgb', 'flow']:
        raise ValueError('{} not among known modalities [rgb|flow]'.format(modality))
    print('loading data')
    im_size = 224
    dataset = tv.datasets.ImageFolder(
        ct.I3D_PREPARE_DATASET.as_posix(),
        tv.transforms.Compose([
            tv.transforms.CenterCrop(im_size),
            tv.transforms.ToTensor(),
        ]))
    print('loaded data')
    # Initialize input params
    if modality == 'rgb':
        in_channels = 3
    else:
        in_channels = 2
    frame_nb = 16  # Number of items in depth (temporal) dimension
    class_nb = 400

    # Initialize dataset
    loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print('loaded loader')
    # Initialize PyTorch I3D
    i3dpt = _i3dpt.I3D(num_classes=400, modality=modality)
    print('loaded pytorch model')
    # Initialize TensorFlow I3D
    if modality == 'rgb':
        scope = 'RGB'
    else:
        scope = 'Flow'

    with tf.variable_scope(scope):
        rgb_model = _i3dtf.I3D(class_nb, final_endpoint='Predictions')
        # TensorFlow forward pass
        rgb_input = tf.placeholder(tf.float32, shape=(batch_size, frame_nb, im_size, im_size, in_channels))
        rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

    # Get params for TensorFlow weight retrieval
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == scope:
            rgb_variable_map[variable.name.replace(':0', '')] = variable

    criterion = th.nn.L1Loss()
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    with tf.Session() as sess:
        # Load saved TensorFlow weights
        rgb_saver.restore(sess, tf_checkpoint)

        # Transfer weights from TensorFlow to pytorch
        i3dpt.eval()
        # i3dpt.load_tf_weights(sess)

        # Save PyTorch weights for future loading
        th.save(i3dpt.cpu().state_dict(), pt_checkpoint)

        # Load data
        for i, (input_2d, target) in enumerate(loader):
            if modality == 'flow':
                input_2d = input_2d[:, 0:2]  # Remove one dimension

            # Prepare data for pytorch forward pass
            input_3d = input_2d.clone().unsqueeze(2).repeat(1, 1, frame_nb, 1, 1)
            input_3d_pt = input_3d.clone()

            # Prepare data for TensorFlow pass
            feed_dict = {}
            input_3d_tf = input_3d.numpy().transpose(0, 2, 3, 4, 1)
            feed_dict[rgb_input] = input_3d_tf

            # TensorFlow forward pass
            tf_out3d_sample = sess.run(rgb_logits, feed_dict=feed_dict)
            out_tf_np = tf_out3d_sample

            # Pytorch forward pass
            out_pt, _ = i3dpt(input_3d_pt)
            out_pt_np = out_pt.data.numpy()
            loss = criterion(out_pt, th.ones_like(out_pt))
            # Pytorch backward pass
            loss.backward()

            # Make sure the TensorFlow and PyTorch outputs have the same shape
            assert out_tf_np.shape == out_pt_np.shape, f'tf output: {out_tf_np.shape} != pt output : {out_pt_np.shape}'
            hp.compare_outputs(out_tf_np, out_pt_np)
