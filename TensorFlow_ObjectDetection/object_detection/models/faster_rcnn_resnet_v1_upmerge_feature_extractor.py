# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Resnet V1 Faster R-CNN implementation.

See "Deep Residual Learning for Image Recognition" by He et al., 2015.
https://arxiv.org/abs/1512.03385

Note: this implementation assumes that the classification checkpoint used
to finetune this model is trained using the same configuration as that of
the MSRA provided checkpoints
(see https://github.com/KaimingHe/deep-residual-networks), e.g., with
same preprocessing, batch norm scaling, etc.
"""
import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import resnet_utils
from nets import resnet_v1

slim = tf.contrib.slim


class FasterRCNNResnetV1UpsampleFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Resnet V1 feature extractor implementation."""

  def __init__(self,
               architecture,
               resnet_model,
               is_training,
               first_stage_features_stride,
               first_stage_features_outblock,
               second_stage_features_depth, 
               second_stage_head_sel,
               use_avgpool, 
               batch_norm_trainable=False,
               reuse_weights=None,
               # second_stage_head_sel,
               weight_decay=0.0):
    """Constructor.

    Args:
      architecture: Architecture name of the Resnet V1 model.
      resnet_model: Definition of the Resnet V1 model.
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride >> 3 not in [1, 2, 4, 8]:
      # first_stage_features_stride could be in [8, 16, 32, 64] with resnet.
      raise ValueError('`first_stage_features_stride` must be a `8 x 2^n` value.')
    if (first_stage_features_outblock == 3 and first_stage_features_stride > 32) or (first_stage_features_outblock == 4 and first_stage_features_stride > 64):
      raise ValueError('`first_stage_features_stride` outranges the output conv-level.')
    # if first_stage_features_stride != 8 and first_stage_features_stride != 16:
    #   raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._architecture = architecture
    self._resnet_model = resnet_model
    super(FasterRCNNResnetV1UpsampleFeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, 
        first_stage_features_outblock, 
        second_stage_features_depth, 
        second_stage_head_sel, use_avgpool, 
        batch_norm_trainable,
        # second_stage_head_sel, 
        reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    channel_means = [123.68, 116.779, 103.939]
    return resized_inputs - [[channel_means]]

  def _filter_features(self, image_features):
    # TODO(rathodv): Change resnet endpoint to strip scope prefixes instead
    # of munging the scope here.
    filtered_image_features = dict({})
    for key, feature in image_features.items():
      feature_name = key.split('/')[-1]
      if feature_name in ['block1', 'block2', 'block3', 'block4']:
        filtered_image_features[feature_name] = feature
    return filtered_image_features

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
            tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      # Disables batchnorm for fine-tuning with smaller batch sizes.
      # TODO(chensun): Figure out if it is needed when image
      # batch size is bigger.
      with slim.arg_scope(
          resnet_utils.resnet_arg_scope(
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              weight_decay=self._weight_decay)):
        with tf.variable_scope(
            self._architecture, reuse=self._reuse_weights) as var_scope:
          _, image_features = self._resnet_model(
              preprocessed_inputs,
              num_classes=None,
              is_training=self._train_batch_norm,
              global_pool=False,
              output_stride=self._first_stage_features_stride,
              spatial_squeeze=False,
              scope=var_scope)

          block_name = ['block3', 'block4'] if self._first_stage_features_outblock == 4 else ['block2', 'block3']
          features_depth_output = 512
          image_features = self._filter_features(image_features)
          last_feature_map = image_features[block_name[1]]
          last_feature_map = slim.conv2d(
              last_feature_map,
              num_outputs=features_depth_output,
              kernel_size=[1, 1],
              stride=1,
              activation_fn=None,
               normalizer_fn=None,
              padding='SAME',
              scope=block_name[1]+'_side')
          second_last_feature_map = image_features[block_name[0]]
          top_down = tf.image.resize_images(last_feature_map, tf.shape(second_last_feature_map)[1:3])
          residual = slim.conv2d(
               second_last_feature_map,
               num_outputs =features_depth_output,
               kernel_size=[1, 1],
               stride = 1,
               activation_fn=None,
               normalizer_fn=None,
               padding='SAME',
               scope=block_name[0]+'_side')
          top_down = 0.5 * top_down + 0.5 * residual
          # top_down_up = tf.image.resize_images(top_down, tf.shape(image_features['block1'])[1:3])
          # residual_1 = slim.conv2d(
          #      image_features['block1'],
          #      num_outputs = 256,
          #      kernel_size=[1, 1],
          #      stride = 1,
          #      activation_fn=None,
          #      normalizer_fn=None,
          #      padding='SAME',
          #      scope='block1_side')
          # top_down = 0.5 * top_down_up + 0.5 * residual_1
    return top_down, image_features

    # block_name = 'block3' if self._first_stage_features_outblock == 3 else 'block4'
    # handle = scope + '/%s/' % self._architecture + block_name
    # return activations[handle], activations

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    if self._second_stage_head_sel == 'res_block4' and self._first_stage_features_outblock == 4:
      raise ValueError('first_stage_features outs from res-block4, should use other head!')
    
    if self._second_stage_head_sel == 'res_block4':
      with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
        with slim.arg_scope(
            resnet_utils.resnet_arg_scope(
                batch_norm_epsilon=1e-5,
                batch_norm_scale=True,
                weight_decay=self._weight_decay)):
          with slim.arg_scope([slim.batch_norm],
                              is_training=self._train_batch_norm):
            blocks = [
                resnet_utils.Block('block4', resnet_v1.bottleneck, [{
                    'depth': self._second_stage_features_depth,
                    'depth_bottleneck': 512,
                    'stride': 1
                }] * 3)
            ]
            proposal_classifier_features = resnet_utils.stack_blocks_dense(
                proposal_feature_maps, blocks)
      return proposal_classifier_features
    
    if self._second_stage_head_sel == 'light_head':
      # pass
      return proposal_classifier_features
    
    if self._second_stage_head_sel == 'direct_fc':
      # proposal_feature_maps = tf.reduce_mean(proposal_feature_maps, [1, 2], keep_dims=True, name='AvgPool')
      with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
        if self._use_avgpool:
          proposal_feature_maps = tf.reduce_mean(proposal_feature_maps, [1, 2], keep_dims=True, name='RoIPool_AvgPool')
        with slim.arg_scope([slim.fully_connected],
                             weights_regularizer=slim.l2_regularizer(self._weight_decay),
                             normalizer_fn=slim.batch_norm):
          with slim.arg_scope([slim.batch_norm],
                              is_training=self._train_batch_norm): 
            proposal_classifier_features = slim.fully_connected(
                                                   proposal_feature_maps, 
                                                   num_outputs=self._second_stage_features_depth, 
                                                   trainable=self._train_batch_norm)
      return proposal_classifier_features
    
    # return proposal_classifier_features


class FasterRCNNResnet50UpsampleFeatureExtractor(FasterRCNNResnetV1UpsampleFeatureExtractor):
  """Faster R-CNN Resnet 50 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               first_stage_features_outblock, 
               second_stage_features_depth, 
               second_stage_head_sel,
               use_avgpool, 
               batch_norm_trainable=False,
               reuse_weights=None,
               # second_stage_head_sel,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNResnet50UpsampleFeatureExtractor, self).__init__(
        'resnet_v1_50', resnet_v1.resnet_v1_50, is_training,
        first_stage_features_stride, 
        first_stage_features_outblock, 
        second_stage_features_depth,
        second_stage_head_sel, use_avgpool, 
        batch_norm_trainable,
        reuse_weights,
        # second_stage_head_sel,
        weight_decay)


class FasterRCNNResnet101UpsampleFeatureExtractor(FasterRCNNResnetV1UpsampleFeatureExtractor):
  """Faster R-CNN Resnet 101 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               first_stage_features_outblock,
               second_stage_features_depth, 
               second_stage_head_sel,
               use_avgpool, 
               batch_norm_trainable=False,
               reuse_weights=None,
               # second_stage_head_sel,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNResnet101UpsampleFeatureExtractor, self).__init__(
        'resnet_v1_101', resnet_v1.resnet_v1_101, is_training,
        first_stage_features_stride, 
        first_stage_features_outblock, 
        second_stage_features_depth, 
        second_stage_head_sel, use_avgpool, 
        batch_norm_trainable,
        reuse_weights, 
        # second_stage_head_sel,
        weight_decay)


class FasterRCNNResnet152UpsampleFeatureExtractor(FasterRCNNResnetV1UpsampleFeatureExtractor):
  """Faster R-CNN Resnet 152 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               first_stage_features_outblock,
               second_stage_features_depth, 
               second_stage_head_sel,
               use_avgpool, 
               batch_norm_trainable=False,
               reuse_weights=None,
               # second_stage_head_sel,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNResnet152UpsampleFeatureExtractor, self).__init__(
        'resnet_v1_152', resnet_v1.resnet_v1_152, is_training,
        first_stage_features_stride, 
        first_stage_features_outblock, 
        second_stage_features_depth, 
        second_stage_head_sel, use_avgpool, 
        batch_norm_trainable, 
        reuse_weights,
        # second_stage_head_sel,
        weight_decay)
