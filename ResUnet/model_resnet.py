import tensorflow as tf
from Vnet.layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add,
                        weight_xavier_init, bias_variable, save_images)
import numpy as np


def nor_data(images):
    max = np.max(images)
    min = np.min(images)
    new_images = (images-min)/(max-min)
    return new_images


def get_denoise_data(batch_info, pos_data):
    """
    load LD_data and HD_data
    :param batch_info: [num,z,x,y]
    :param pos_data: [[num,start,end], ...]
    :return:
    """
    num = str(batch_info[0])
    file_path = os.path.join(r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/denoise/train', num, 'HDCT')
    HD_list = os.listdir(file_path)
    length = len(HD_list)
    for i in range(len(pos_data)):
        if str(pos_data[i][0]) == num:
            start = pos_data[i][1]
            break
    HD_images = np.zeros((length, 512, 512), dtype=np.float32)
    for i in range(length):
        slice = int(HD_list[i].split('_')[2].split('.')[0])-1
        img = np.fromfile(file_path+'/'+HD_list[i], dtype='float32')
        img = img.reshape((512, 512))
        HD_images[slice, :, :] = img
    file_path = os.path.join(r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/denoise/train', num, 'LDCT')
    LD_list = os.listdir(file_path)
    length = len(LD_list)
    for i in range(len(pos_data)):
        if str(pos_data[i][0]) == num:
            start = pos_data[i][1]
            break
    LD_images = np.zeros((length, 512, 512), dtype=np.float32)
    for i in range(length):
        slice = int(LD_list[i].split('_')[2].split('.')[0])-1
        img = np.fromfile(file_path+'/'+LD_list[i], dtype='float32')
        img = img.reshape((512, 512))
        HD_images[slice, :, :] = img
    ld_img = np.zeros((16, 256, 256), dtype=np.float32)
    hd_img = np.zeros((16, 256, 256), dtype=np.float32)
    if length >= start+batch_info[1]+16:
        ld_img[:, :, :] = HD_images[start+batch_info[1]:start+batch_info[1]+16, batch_info[2]:batch_info[2]+256,
                       batch_info[3]:batch_info[3]+256]
        hd_img[:, :, :] = LD_images[start+batch_info[1]:start+batch_info[1]+16, batch_info[2]:batch_info[2]+256,
                   batch_info[3]:batch_info[3]+256]
    else:
        ld_img[:, :, :] = HD_images[length-16:length, batch_info[2]:batch_info[2]+256,
                       batch_info[3]:batch_info[3]+256]
        hd_img[:, :, :] = LD_images[length-16:length, batch_info[2]:batch_info[2]+256,
                   batch_info[3]:batch_info[3]+256]
    return ld_img, hd_img


def conv_bn_relu_drop(x, kernel, phase, drop, depth=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2] * kernel[3],
                               n_outputs=kernel[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernel[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def down_sampling(x, kernel, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2] * kernel[3],
                               n_outputs=kernel[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernel[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def deconv_relu(x, kernel, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2] * kernel[-1],
                               n_outputs=kernel[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernel[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.relu(conv)
        return conv


def conv_output(x, kernel, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernel, n_inputs=kernel[0] * kernel[1] * kernel[2] * kernel[3],
                               n_outputs=kernel[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernel[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        return conv


def _create_conv_net(X, image_depth, image_height, image_width, image_channel, inner_channel, phase, drop):
    inputX = tf.reshape(X, [-1, image_depth, image_width, image_height, image_channel])  # shape=(?, 256, 256, 1)
    # ResUnet model
    # Res-block0
    layer0_0 = conv_bn_relu_drop(x=inputX, kernel=(3, 3, 3, image_channel, inner_channel), phase=phase, drop=drop,
                                 scope='layer0_0')
    layer0_1 = conv_bn_relu_drop(x=layer0_0, kernel=(3, 3, 3, inner_channel, inner_channel), phase=phase, drop=drop,
                                 scope='layer0_1')
    layer0_2 = conv_bn_relu_drop(x=layer0_0, kernel=(3, 3, 3, inner_channel, inner_channel), phase=phase, drop=drop,
                                 scope='layer0_2')
    layer0 = resnet_Add(x1=layer0_1, x2=layer0_2)
    # down sampling
    down0 = down_sampling(x=layer0, kernel=(3, 3, 3, inner_channel, 2*inner_channel), phase=phase, drop=drop,
                                 scope='down0')

    # Res-block1
    layer1_1 = conv_bn_relu_drop(x=down0, kernel=(3, 3, 3, 2*inner_channel, 2*inner_channel), phase=phase, drop=drop,
                                 scope='layer1_1')
    layer1_2 = conv_bn_relu_drop(x=layer1_1, kernel=(3, 3, 3, 2*inner_channel, 2*inner_channel), phase=phase, drop=drop,
                                 scope='layer1_2')
    layer1 = resnet_Add(x1=down0, x2=layer1_2)
    # down sampling
    down1 = down_sampling(x=layer1, kernel=(3, 3, 3, 2*inner_channel, 4*inner_channel), phase=phase, drop=drop,
                                 scope='down1')

    # Res-block2
    layer2_1 = conv_bn_relu_drop(x=down1, kernel=(3, 3, 3, 4*inner_channel, 4*inner_channel), phase=phase, drop=drop,
                                 scope='layer2_1')
    layer2_2 = conv_bn_relu_drop(x=layer2_1, kernel=(3, 3, 3, 4*inner_channel, 4*inner_channel), phase=phase, drop=drop,
                                 scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2_2)
    # down sampling
    down2 = down_sampling(x=layer2, kernel=(3, 3, 3, 4*inner_channel, 8*inner_channel), phase=phase, drop=drop,
                                 scope='down2')

    # Res-block3
    layer3_1 = conv_bn_relu_drop(x=down2, kernel=(3, 3, 3, 8*inner_channel, 8*inner_channel), phase=phase, drop=drop,
                                 scope='layer3_1')
    layer3_2 = conv_bn_relu_drop(x=layer3_1, kernel=(3, 3, 3, 8*inner_channel, 8*inner_channel), phase=phase, drop=drop,
                                 scope='layer3_2')
    layer3 = resnet_Add(x1=down2, x2=layer3_2)

    # up sampling
    up2 = deconv_relu(x=layer3, kernel=(3, 3, 3, 4*inner_channel, 8*inner_channel), scope='up3')
    # Res-block4
    concat0 = crop_and_concat(layer2, up2)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer4_1 = conv_bn_relu_drop(x=concat0, kernel=(3, 3, 3, 8*inner_channel, 4*inner_channel),
                                 depth=Z, height=H, width=W, phase=phase, drop=drop, scope='layer4_1')
    layer4_2 = conv_bn_relu_drop(x=layer4_1, kernel=(3, 3, 3, 4*inner_channel, 4*inner_channel),
                                 depth=Z, height=H, width=W, phase=phase, drop=drop, scope='layer4_2')
    layer4 = resnet_Add(x1=up2, x2=layer4_2)

    # up sampling
    up1 = deconv_relu(x=layer4, kernel=(3, 3, 3, 2*inner_channel, 4*inner_channel), scope='up3')
    # Res-block5
    concat1 = crop_and_concat(layer1, up1)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer5_1 = conv_bn_relu_drop(x=concat1, kernel=(3, 3, 3, 4*inner_channel, 2*inner_channel),
                                 depth=Z, height=H, width=W, phase=phase, drop=drop, scope='layer4_1')
    layer5_2 = conv_bn_relu_drop(x=layer5_1, kernel=(3, 3, 3, 2*inner_channel, 2*inner_channel),
                                 depth=Z, height=H, width=W, phase=phase, drop=drop, scope='layer4_2')
    layer5 = resnet_Add(x1=up1, x2=layer5_2)

    # up sampling
    up0 = deconv_relu(x=layer5, kernel=(3, 3, 3, inner_channel, 2*inner_channel), scope='up3')
    # Res-block6
    concat2 = crop_and_concat(layer0, up0)
    _, Z, H, W, _ = layer0.get_shape().as_list()
    layer6_1 = conv_bn_relu_drop(x=concat2, kernel=(3, 3, 3, 2*inner_channel, inner_channel),
                                 depth=Z, height=H, width=W, phase=phase, drop=drop, scope='layer4_1')
    layer6_2 = conv_bn_relu_drop(x=layer6_1, kernel=(3, 3, 3, inner_channel, inner_channel),
                                 depth=Z, height=H, width=W, phase=phase, drop=drop, scope='layer4_2')
    layer6 = resnet_Add(x1=up0, x2=layer6_2)

    output = conv_output(layer6, kernel=(3, 3, 3, inner_channel, 1), scope='output')
    return output


# Serve data by batches
def _next_batch(train_images, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_images[start:end], index_in_epoch


class ResUnet3dModule(object):
    def __init__(self, image_height, image_width, image_depth, channels=1, inner_channel=16, costname=("mse",),
                 inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.channels = channels
        self.inner_channel = inner_channel

        self.X = tf.placeholder("float", shape=[None, self.image_depth, self.image_height, self.image_width,
                                                self.channels])
        self.Y_gt = tf.placeholder("float", shape=[None, self.image_depth, self.image_height, self.image_width,
                                                   self.channels])
        self.lr = tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)
        self.drop = tf.placeholder('float')
        self.Y_pred = _create_conv_net(self.X, self.image_depth, self.image_width, self.image_height, self.channels,
                                       self.inner_channel, self.phase, self.drop)
        self.cost = self.__get_cost(costname[0])
        self.accuracy = -self.__get_cost(costname[0])
        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)

    def __get_cost(self, cost_name):
        Z, H, W, C = self.Y_gt.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            pred_flat = tf.reshape(self.Y_pred, [-1, H * W * C * Z])
            true_flat = tf.reshape(self.Y_gt, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
        if cost_name == "mse":
            pred_flat = tf.squeeze(self.Y_pred)
            label_flat = tf.squeeze(self.Y_gt)
            loss = tf.reduce_mean(tf.square(label_flat - pred_flat))
        return loss

    def train(self, train_images, pos_data, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=10, batch_size=1):
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(logs_path + "model/"):
            os.makedirs(logs_path + "model/")
        model_path = logs_path + "model/" + model_path
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        DISPLAY_STEP = 1
        index_in_epoch = 0

        train_epochs = train_images.shape[0] * train_epochs
        for i in range(train_epochs):
            # get new batch
            batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(train_images, batch_size,
                                                                       index_in_epoch)
            batch_xs = np.empty((len(batch_xs_path), self.image_depth, self.image_height, self.image_width,
                                 self.channels))
            batch_ys = np.empty((len(batch_ys_path), self.image_depth, self.image_height, self.image_width,
                                 self.channels))
            for num in range(len(batch_xs_path)):
                image, label = get_denoise_data(batch_xs_path[num], pos_data)
                batch_xs[num, :, :, :, :] = np.reshape(image, (self.image_depth, self.image_height,
                                                               self.image_width, self.channels))
                batch_ys[num, :, :, :, :] = np.reshape(label, (self.image_depth, self.image_height,
                                                               self.image_width, self.channels))
            # train on batch
            print("[Batch %d/%d]" % (i, train_epochs))
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop: dropout_conv})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, test_images):
        test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
        test_images = test_images.astype(np.float)
        y_dummy = test_images
        pred = self.sess.run(self.Y_pred, feed_dict={self.X: [test_images],
                                                     self.Y_gt: [y_dummy],
                                                     self.phase: 1,
                                                     self.drop: 1})
        # normalization
        pred = nor_data(pred)
        result = pred.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        result = np.reshape(result, (test_images.shape[0], test_images.shape[1], test_images.shape[2]))
        return result
