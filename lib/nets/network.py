# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np
import scipy
from collections import Counter
import random
from copy import deepcopy

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from layer_utils.idn_proposal_layer import idn_qual_proposal_layer, idn_sim_proposal_layer, idn_proposal_test_layer, idn_sim_target_layer
from utils.visualization import draw_bounding_boxes
from utils.tf_utils import zerof, instead, cbody, mbody, whilef, concat

from model.config import cfg
from collections import deque
MEMORY_SIZE = 10

class Network(object):
  def __init__(self):
    self._predictions = {}
    self._idn_sim_values = {}
    self._idn_qual_values = {}
    self._idn_values = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}

    # self._rin_placeholder = {
    #   "input_boxes": tf.placeholder(tf.float32, [None, 5]),
    #   "idn_net_conv": tf.placeholder(tf.int32, [None, None])
    # }
    self._rin_placeholder = {
      "feat_cropped": tf.placeholder(tf.float32, [None, 31, 31, 128]),
      "pIOU": tf.placeholder(tf.float32, [None, None]),
      "dppLabel": tf.placeholder(tf.int32, [None]),
      "intraDppLabel": tf.placeholder(tf.int32, [None, None]),
      "clssLabel": tf.placeholder(tf.int32, [None, None])
    }
    self._layers = {}
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}
    self._memory = deque(maxlen=MEMORY_SIZE)

  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._im_info],
                      tf.float32, name="gt_boxes")
    
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:
        rois, rpn_scores = proposal_top_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
      else:
        rois, rpn_scores = tf.py_func(proposal_top_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal_top")
        
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:
        rois, rpn_scores = proposal_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._mode,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
      else:
        rois, rpn_scores = tf.py_func(proposal_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal")

      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, pre_pool_size, feat_stride, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(feat_stride)
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(feat_stride)
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = pre_pool_size * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      self._score_summaries.update(self._proposal_targets)

      return rois, roi_scores

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      if cfg.USE_E2E_TF:
        anchors, anchor_length = generate_anchors_pre_tf(
          height,
          width,
          self._feat_stride,
          self._anchor_scales,
          self._anchor_ratios
        )
      else:
        anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                            [height, width,
                                             self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                            [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _build_network(self):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    net_conv, idn_net_conv = self._image_to_head(self.frcnn_training or self.quality_training)
    self.idn_net_conv = idn_net_conv
    with tf.variable_scope(self._scope, self._scope):
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      rois = self._region_proposal(net_conv, self.frcnn_training or self.quality_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv, rois, 7, 16, "pool5")
        # pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError
    fc7 = self._head_to_tail(pool5, self.frcnn_training or self.quality_training)
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, self.frcnn_training or self.quality_training,
                                                        initializer, initializer_bbox)

    if self.frcnn_training or self.testing_nms:
      self._score_summaries.update(self._predictions)
      return rois, cls_prob, bbox_pred

    else:
        # intra-inter class dpp network.
        if self.testing_dpp:
          input_boxes, clss, input_clss, input_scores = self._idn_proposal_test_layer(cls_prob,
                                                                                      bbox_pred,
                                                                                      rois,
                                                                                      "idn")
          net = self._crop_pool_layer(idn_net_conv, input_boxes, 31, 4, "pool")
          idn_feat_sim = self._rin(net, False)
          self._predictions["idn_feat_sim"] = idn_feat_sim

        elif self.quality_training:
          input_boxes, clss, input_clss, input_scores = self._idn_qual_proposal_layer(self._predictions['cls_prob'],
                                                                                      self._predictions['bbox_pred'],
                                                                                      self._gt_boxes,
                                                                                      self._predictions['rois'],
                                                                                      "idn")
          net = self._crop_pool_layer(idn_net_conv, input_boxes, 31, 4, "pool")
          idn_feat_qual = self._rin(net, True)
          self._predictions["idn_feat_qual"] = idn_feat_qual

        elif self.similarity_training:
          # input_boxes, input_clss, input_scores = self._idn_sim_proposal_layer(self._rin_placeholder['cls_prob'],
          #                                                                      self._rin_placeholder['bbox_pred'],
          #                                                                      self._gt_boxes,
          #                                                                      self._rin_placeholder['rois'],
          #                                                                      "idn")
          # input_boxes, input_clss, input_scores = self._idn_sim_proposal_layer(self._predictions['cls_prob'],
          #                                                                      self._predictions['bbox_pred'],
          #                                                                      self._gt_boxes,
          #                                                                      self._predictions['rois'],
          #                                                                      "idn")
          feat_cropped = self._crop_pool_layer(self._rin_placeholder["idn_net_conv"], self._rin_placeholder['input_boxes'], 31, 4, "pool")
          idn_feat_sim = self._rin(feat_cropped, True)
          self._predictions["idn_feat_sim"] = idn_feat_sim
        self._score_summaries.update(self._predictions)
        return rois

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def _SS_loss(self, idn_feat_sim, pIOU, posDppLabel, quality):
    self.V = V = tf.nn.l2_normalize(idn_feat_sim, 1)
    self.S = S = tf.matmul(V, V, False, True)
    M = tf.shape(V)[0]
    L = tf.sqrt(tf.tile(quality, [1, M])) * (0.6*S + 0.4*pIOU) * tf.sqrt(tf.tile(tf.transpose(quality), [M, 1]))
    L = (L + tf.transpose(L))/2.

    # SS Loss : Decrease the score of subsets that contain at least one bad bounding boxes.
    denom = tf.square(tf.reduce_prod(tf.diag_part(tf.cholesky(L + tf.eye(tf.size(L[0]))))))
    p_nom = tf.square(tf.reduce_prod(tf.diag_part(tf.cholesky(tf.gather(tf.transpose(tf.gather(L+tf.eye(tf.size(L[0])),posDppLabel)),posDppLabel)))))
    pos_loss = -tf.log(p_nom) + tf.log(denom)
    loss = pos_loss
    except_loss = tf.log(tf.square(tf.reduce_prod(
        tf.diag_part(tf.cholesky(tf.gather(tf.transpose(tf.gather(L + tf.eye(tf.size(L[0])), posDppLabel)), posDppLabel))))))

    # Give temporary loss
    loss = tf.cond(tf.is_nan(loss) | tf.is_inf(loss),
            lambda: instead(except_loss),
            lambda: instead(loss))
    return loss

  def _ID_loss(self, idn_feat_sim, pIOU, intraDppLabel, clssLabel, dppLabel, quality): #
    self.V = V = tf.nn.l2_normalize(idn_feat_sim, 1)
    self.S = S = tf.matmul(V, V, False, True)
    M = tf.shape(V)[0]
    L = tf.sqrt(tf.tile(quality, [1, M])) * (0.6*S + 0.4*pIOU) * tf.sqrt(tf.tile(tf.transpose(quality), [M, 1]))
    L = (L + tf.transpose(L))/2.
    denom = tf.square(tf.reduce_prod(tf.diag_part(tf.cholesky(L + tf.eye(tf.size(L[0]))))))
    nom = tf.reduce_prod(tf.gather(quality, dppLabel))*tf.square(tf.reduce_prod(tf.diag_part(tf.cholesky(tf.gather(tf.transpose(tf.gather(S + cfg.EPSILON*tf.eye(tf.size(L[0])), dppLabel)), dppLabel)))))
    inter_loss = -(tf.log(nom) - tf.log(denom))
    inter_except_loss = tf.square(tf.reduce_prod(tf.diag_part(tf.cholesky(tf.gather(tf.transpose(tf.gather(L + tf.eye(tf.size(L[0])), dppLabel)), dppLabel)))))

    inter_loss = tf.cond(tf.is_nan(inter_loss) | tf.is_inf(inter_loss),
            lambda: instead(inter_except_loss),
            lambda: instead(inter_loss))

    intra_clss_index = (tf.constant(0), tf.constant(0.0), clssLabel, L)
    intra_denom = tf.divide(whilef(intra_clss_index, mbody), tf.cast(tf.shape(clssLabel)[0], tf.float32))
    intra_label_index = (tf.constant(0), tf.constant(0.0), intraDppLabel, L)
    intra_nom = tf.divide(whilef(intra_label_index, cbody), tf.cast(tf.shape(intraDppLabel)[0], tf.float32))

    intra_loss = -(tf.log(intra_nom) - tf.log(intra_denom))
    intra_except_loss = tf.divide(whilef(intra_label_index, mbody), tf.cast(tf.shape(intraDppLabel)[0], tf.float32))
    intra_loss = tf.cond(tf.is_nan(intra_loss) | tf.is_inf(intra_loss),
             lambda: instead(intra_except_loss),
             lambda: instead(intra_loss))

    loss = intra_loss + inter_loss
    # loss = inter_loss
    return loss

  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      if self.quality_training:
        # RPN, class loss
        rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
        rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # RPN, bbox loss
        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

        # RCNN, class loss
        cls_score = self._predictions["cls_score"]
        label = tf.reshape(self._proposal_targets["labels"], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # RCNN, bbox loss
        bbox_pred = self._predictions['bbox_pred']
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box
        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box
        frcnn_loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # IDN, dpp loss
        SS_loss = tf.cond(self._idn_qual_values['idn_train'],
                lambda: self.train_func(),
                lambda: zerof())
        self._losses['SS_loss'] = SS_loss
        loss = SS_loss + frcnn_loss

      elif self.similarity_training:
        # IDN, dpp loss
        ID_loss = self.train_sim_func()
        # ID_loss = tf.cond(self._idn_sim_values['idn_train'],
        #         lambda: self.train_sim_func(),
        #         lambda: zerof())
        self._losses['ID_loss'] = ID_loss
        loss = ID_loss

      elif self.frcnn_training:
        # RPN, class loss
        rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
        rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # RPN, bbox loss
        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

        # RCNN, class loss
        cls_score = self._predictions["cls_score"]
        label = tf.reshape(self._proposal_targets["labels"], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # RCNN, bbox loss
        bbox_pred = self._predictions['bbox_pred']
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box
        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

      else:
        print ("error")
        loss = 0

      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

      self._event_summaries.update(self._losses)
    return loss

  def train_func(self):
    idn_feat_qual = self._predictions['idn_feat_qual']
    pIOU = self._idn_qual_values['pIOU']
    posDppLabel = self._idn_qual_values['posDppLabel']
    add_gt = self._idn_qual_values['add_gt']
    masked_quality = tf.boolean_mask(self._predictions['dpp_quality'], self._idn_qual_values['mask'])
    quality = tf.expand_dims(tf.cond(add_gt, lambda: concat(masked_quality, tf.ones([tf.shape(self._gt_boxes)[0]]) * 4),
                                     lambda: instead(masked_quality)), -1)
    SS_loss = cfg.TRAIN.QUAL_LR * self._SS_loss(idn_feat_qual, pIOU, posDppLabel, quality)
    return SS_loss

  def train_sim_func(self):
    idn_feat_sim = self._predictions["idn_feat_sim"]
    clssLabel = self._rin_placeholder['clssLabel']
    pIOU = self._rin_placeholder['pIOU']
    intraDppLabel = self._rin_placeholder['intraDppLabel']
    dppLabel = self._rin_placeholder['dppLabel']
    quality = tf.ones([tf.shape(idn_feat_sim)[0], 1], tf.float32)
    ID_loss = cfg.TRAIN.SIM_LR * self._ID_loss(idn_feat_sim, pIOU, intraDppLabel, clssLabel, dppLabel, quality) #
    return ID_loss

  def _region_proposal(self, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    self._act_summaries.append(rpn)
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois

  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    dpp_quality = 3.75*(cls_prob-tf.reduce_min(cls_prob))/(tf.reduce_max(cls_prob)-tf.reduce_min(cls_prob))+0.25
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["dpp_quality"] = dpp_quality

    return cls_prob, bbox_pred

  def _idn_qual_proposal_layer(self, cls_score, bbox_pred, gts, rois, name):
    stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
    means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
    bbox_pred *= stds
    bbox_pred += means
    with tf.variable_scope(name) as scope:
      input_boxes, input_clss, input_scores, pIOU, clss, objects, \
      idn_train, mask, add_gt, posDppLabel \
        = tf.py_func(idn_qual_proposal_layer,
                     [cls_score, bbox_pred, gts, rois, self._im_info, self._num_classes],
                     [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32,
                      tf.int32, tf.int32, tf.bool, tf.int32],
                     name="idn_proposal")
      input_boxes = tf.reshape(input_boxes, [-1, 5], name='input_boxes')
      input_clss = tf.reshape(input_clss, [-1, 1], name='input_clss')
      input_scores = tf.reshape(input_scores, [-1, ], name='input_scores')
      pIOU = tf.reshape(pIOU, [tf.shape(pIOU)[0], tf.shape(pIOU)[1]], name='pIOU')
      idn_train = tf.cast(tf.reshape(idn_train,[]), tf.bool)
      clss = tf.reshape(clss, [-1, self._num_classes], name='clss')
      posDppLabel = tf.reshape(posDppLabel, [-1, ], name='posDppLabel')
      mask = tf.cast(tf.reshape(mask, [tf.shape(mask)[0], self._num_classes], name='mask'), tf.bool)
      self.objects = tf.reshape(objects, [-1], name="objects")

    self._idn_qual_values["input_boxes"] = input_boxes
    self._idn_qual_values["input_clss"] = input_clss
    self._idn_qual_values["input_scores"] = input_scores
    self._idn_qual_values["pIOU"] = pIOU
    self._idn_qual_values["idn_train"] = idn_train
    self._idn_qual_values["clss"] = clss
    self._idn_qual_values['posDppLabel'] = posDppLabel
    self._idn_qual_values['mask'] = mask
    self._idn_qual_values['add_gt'] = add_gt

    return input_boxes, clss, input_clss, input_scores

  def _idn_sim_proposal_layer(self, cls_score, bbox_pred, gts, rois, name):
    stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
    means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
    bbox_pred *= stds
    bbox_pred += means
    with tf.variable_scope(name) as scope: #idn_train, dppLabel, mask, add_gt , clss, objects\
      input_boxes, input_clss, input_scores, pIOU, dppLabel, intraDppLabel, clssLabel \
        = tf.py_func(idn_sim_proposal_layer,
                     [cls_score, bbox_pred, gts, rois, self._im_info, self._num_classes],
                     [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32], name="idn_proposal")
      # tf.int32 tf.int32, tf.int32, tf.int32, tf.bool , tf.float32, tf.int32

      input_boxes = tf.reshape(input_boxes, [-1, 5], name='input_boxes')
      input_clss = tf.reshape(input_clss, [-1, 1], name='input_clss')
      input_scores = tf.reshape(input_scores, [-1, ], name='input_scores')
      pIOU = tf.reshape(pIOU, [tf.shape(pIOU)[0], tf.shape(pIOU)[1]], name='pIOU')
      # idn_train = tf.cast(tf.reshape(idn_train,[]), tf.bool)
      # clss = tf.reshape(clss, [-1, self._num_classes], name='clss')
      dppLabel = tf.reshape(dppLabel, [-1, ], name='dppLabel')
      intraDppLabel = tf.reshape(intraDppLabel, [tf.shape(intraDppLabel)[0], tf.shape(intraDppLabel)[1]], name='intraDppLabel')
      clssLabel = tf.reshape(clssLabel, [tf.shape(clssLabel)[0], tf.shape(clssLabel)[1]], name='clssLabel')
      # mask = tf.cast(tf.reshape(mask, [tf.shape(mask)[0], self._num_classes], name='mask'), tf.bool)
      # self.objects = tf.reshape(objects, [-1], name="objects")

    self._idn_sim_values["input_boxes"] = input_boxes
    self._idn_sim_values["input_clss"] = input_clss
    self._idn_sim_values["input_scores"] = input_scores
    self._idn_sim_values["pIOU"] = pIOU
    # self._idn_sim_values["idn_train"] = idn_train
    # self._idn_sim_values["clss"] = clss
    self._idn_sim_values['intraDppLabel'] = intraDppLabel
    self._idn_sim_values['clssLabel'] = clssLabel
    self._idn_sim_values['dppLabel'] = dppLabel
    # self._idn_sim_values['mask'] = mask
    # self._idn_sim_values['add_gt'] = add_gt

    return input_boxes, input_clss, input_scores

  def _idn_sim_target_layer(self, bbox_pred, gts, name):
    stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
    means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
    bbox_pred *= stds
    bbox_pred += means
    with tf.variable_scope(name) as scope:
      input_boxes, input_clss, pIOU, clss, objects,\
      idn_train, dppLabel, intraDppLabel, clssLabel \
        = tf.py_func(idn_sim_target_layer,
                     [gts, self._im_info, self._num_classes],
                     [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32,
                      tf.int32, tf.int32, tf.int32, tf.int32],
                     name="idn_proposal")

      input_boxes = tf.reshape(input_boxes, [-1, 5], name='input_boxes')
      input_clss = tf.reshape(input_clss, [-1, 1], name='input_clss')
      pIOU = tf.reshape(pIOU, [tf.shape(pIOU)[0], tf.shape(pIOU)[1]], name='pIOU')
      idn_train = tf.cast(tf.reshape(idn_train,[]), tf.bool)
      clss = tf.reshape(clss, [-1, self._num_classes], name='clss')
      dppLabel = tf.reshape(dppLabel, [-1, ], name='dppLabel')
      intraDppLabel = tf.reshape(intraDppLabel, [tf.shape(intraDppLabel)[0], tf.shape(intraDppLabel)[1]], name='intraDppLabel')
      clssLabel = tf.reshape(clssLabel, [tf.shape(clssLabel)[0], tf.shape(clssLabel)[1]], name='clssLabel')
      self.objects = tf.reshape(objects, [-1], name="objects")

    self._idn_sim_values["input_boxes"] = input_boxes
    self._idn_sim_values["input_clss"] = input_clss
    self._idn_sim_values["pIOU"] = pIOU
    self._idn_sim_values["idn_train"] = idn_train
    self._idn_sim_values["clss"] = clss
    self._idn_sim_values['intraDppLabel'] = intraDppLabel
    self._idn_sim_values['clssLabel'] = clssLabel
    self._idn_sim_values['dppLabel'] = dppLabel
    return input_boxes, clss, input_clss

  def _idn_proposal_test_layer(self, cls_prob, bbox_pred, rois, name):
    stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
    means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
    bbox_pred *= stds
    bbox_pred += means
    with tf.variable_scope(name) as scope:
      input_boxes, input_clss, input_scores, pIOU, num_patch, clss, mask\
        = tf.py_func(idn_proposal_test_layer,
                     [cls_prob, bbox_pred, rois, self._im_info, self._num_classes],
                     [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32], name="idn_proposal")

      input_boxes = tf.reshape(input_boxes, [-1, 5], name='input_boxes')
      input_clss = tf.reshape(input_clss, [-1, 1], name='input_clss')
      input_scores = tf.reshape(input_scores, [-1, ], name='input_scores')
      pIOU = tf.reshape(pIOU, [tf.shape(pIOU)[0], tf.shape(pIOU)[1]], name='pIOU')
      num_patch = tf.reshape(num_patch, [-1, ], name='num_patch')
      clss = tf.reshape(clss, [-1, self._num_classes], name='clss')
      mask = tf.cast(tf.reshape(mask, [tf.shape(mask)[0], tf.shape(mask)[1]], name='mask'), tf.bool)

    self._idn_values["input_boxes"] = input_boxes
    self._idn_values["input_clss"] = input_clss
    self._idn_values["qualY"] = input_scores
    self._idn_values["pIOU"] = pIOU
    self._idn_values["num_patch"] = num_patch
    self._idn_values["clss"] = clss
    self._idn_values["mask"] = mask

    return input_boxes, clss, input_clss,input_scores
  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  def _rin(self, net, is_training, reuse=None):
      with tf.variable_scope("idn", "idn", reuse=tf.AUTO_REUSE):
          with arg_scope([slim.conv2d, slim.fully_connected],
                         normalizer_fn=slim.batch_norm,
                         normalizer_params={'is_training': is_training}):
                  net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], trainable=is_training, scope='conv1')
                  net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
                  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=is_training, scope='conv2')
                  net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
                  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=is_training, scope='conv3')
                  net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
                  net_flat = slim.flatten(net, scope='flatten')
                  fc6 = slim.fully_connected(net_flat, 1000, scope='fc6', trainable=is_training)
                  fc7 = slim.fully_connected(fc6, 1000, scope='fc7', trainable=is_training)
          idn_feat_sim = slim.fully_connected(fc7, 256, scope='idn_feat_sim', activation_fn=None, trainable=is_training)
      return idn_feat_sim

  # def _rin(self, idn_net_conv, input_boxes, clss, is_training, reuse=None):
  #   with tf.variable_scope("idn", "idn", reuse=reuse):
  #     net = slim.repeat(idn_net_conv, 2, slim.conv2d, 64, [3, 3],
  #                       trainable=is_training, scope='conv1', normalizer_fn=slim.batch_norm, normalizer_params={'trainable': True, 'is_training':is_training})
  #     net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
  #                       trainable=is_training, scope='conv2', normalizer_fn=slim.batch_norm, normalizer_params={'trainable': True, 'is_training':is_training})
  #     net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
  #     net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3],
  #                       trainable=is_training, scope='conv3', normalizer_fn=slim.batch_norm, normalizer_params={'trainable': True, 'is_training':is_training})
  #     net = self._crop_pool_layer(net, input_boxes, 15, 8, "pool")
  #     net_flat = slim.flatten(net, scope='flatten')
  #     fc6 = slim.fully_connected(net_flat, 1000, scope='fc1', normalizer_fn=slim.batch_norm, normalizer_params={'trainable': True, 'is_training':is_training}, trainable=is_training)
  #     add_info = tf.concat([input_boxes, clss], 1)
  #     net_concat = tf.concat([fc6, add_info], 1)
  #     fc7 = slim.fully_connected(net_concat, 1000, scope='fc2', normalizer_fn=slim.batch_norm, normalizer_params={'trainable': True, 'is_training':is_training}, trainable=is_training)
  #     idn_feat_sim = slim.fully_connected(fc7, 256, scope='idn_feat_sim', activation_fn=None, trainable=is_training)
  #   return idn_feat_sim

  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._tag = tag

    self._num_classes = num_classes
    if (mode == "NMS") or (mode == "DPP"):
      self._mode = "TEST"
    else:
      self._mode = "TRAIN"
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios
    self.frcnn_training = mode == 'FRCNN'
    self.quality_training = mode == 'QUAL'
    self.similarity_training = mode == 'SIM'
    self.testing_nms = mode == 'NMS'
    self.testing_dpp = mode == 'DPP'

    assert tag != None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)):
      if self.frcnn_training or self.testing_nms:
        rois, cls_prob, bbox_pred = self._build_network()
      else:
        rois = self._build_network()

    layers_to_output = {'rois': rois}

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if self.testing_nms or self.testing_dpp:
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      self._add_losses()
      layers_to_output.update(self._losses)

      val_summaries = []
      with tf.device("/cpu:0"):
        val_summaries.append(self._add_gt_image_summary())
        for key, var in self._event_summaries.items():
          val_summaries.append(tf.summary.scalar(key, var))
        for key, var in self._score_summaries.items():
          self._add_score_summary(key, var)
        for var in self._act_summaries:
          self._add_act_summary(var)
        for var in self._train_summaries:
          self._add_train_summary(var)

      self._summary_op = tf.summary.merge_all()
      self._summary_op_val = tf.summary.merge(val_summaries)

    layers_to_output.update(self._predictions)

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  # only useful during testing mode
  def test_image_idn(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}
    input_boxes, input_clss, num_patch, pIOU, idn_feat_sim, dpp_quality, mask = sess.run(
                                                    [self._idn_values["input_boxes"],
                                                     self._idn_values['input_clss'],
                                                     self._idn_values['num_patch'],
                                                     self._idn_values['pIOU'],
                                                     self._predictions['idn_feat_sim'],
                                                     self._predictions['dpp_quality'],
                                                     self._idn_values['mask']],
                                                    feed_dict=feed_dict)
    return input_boxes, input_clss, num_patch, pIOU, idn_feat_sim, dpp_quality[mask]

  def get_summary(self, sess, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary
  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    if self.similarity_training:
      sim_train = True #sess.run(self._idn_sim_values['idn_train'], feed_dict=feed_dict)
      qual_train = False
    elif self.quality_training:
      qual_train = sess.run(self._idn_qual_values['idn_train'], feed_dict=feed_dict)
      sim_train = False
    else:
      qual_train = False
      sim_train = False
    blob = {}

    if self.frcnn_training:
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['total_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        blob['rpn_loss_cls']= rpn_loss_cls
        blob['rpn_loss_box']= rpn_loss_box
        blob['loss_cls']= loss_cls
        blob['loss_box']= loss_box
        blob['loss']= loss

    elif self.quality_training:
      rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, SS_loss, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                          self._losses['rpn_loss_box'],
                                                                                          self._losses['cross_entropy'],
                                                                                          self._losses['loss_box'],
                                                                                          self._losses["SS_loss"],
                                                                                          self._losses['total_loss'],
                                                                                          train_op],
                                                                                         feed_dict=feed_dict)
      blob['rpn_loss_cls'] = rpn_loss_cls
      blob['rpn_loss_box'] = rpn_loss_box
      blob['loss_cls'] = loss_cls
      blob['loss_box'] = loss_box
      blob['SS_loss'] = SS_loss
      blob['loss'] = loss

    elif self.similarity_training:
      idn_conv_feat_val, bbox_pred_val, cls_prob_val, rois_val = sess.run([self.idn_net_conv, self._predictions["bbox_pred"],
                                                                           self._predictions["cls_prob"], self._predictions["rois"]], feed_dict=feed_dict)
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      bbox_pred_val *= stds
      bbox_pred_val += means
      self._memory.append((idn_conv_feat_val, cls_prob_val, bbox_pred_val, blobs['gt_boxes'], blobs['im_info']))
      if len(self._memory) >= MEMORY_SIZE:
        sim_feed_dict = self.balanced_sim_feed()
        ID_loss, _ = sess.run([self._losses["ID_loss"], train_op], feed_dict=sim_feed_dict)
        blob['ID_loss'] = ID_loss
    return blob, sim_train, qual_train

  def balanced_sim_feed(self):
    input_boxes, input_clss, input_scores, pIOU, dppLabel, intraDppLabel, clssLabel = idn_sim_proposal_layer(cls_prob_val,
                                                                                                             bbox_pred_val,
                                                                                                             blobs['gt_boxes'],
                                                                                                             rois_val,
                                                                                                             blobs['im_info'],
                                                                                                             self._num_classes)
    idn_conv_feat_val, cls_prob_val, bbox_pred_val, blobs['gt_boxes'], blobs['im_info']
    feat_cropped_val_all, pIOU_all, dppLabel_all, input_clss_all, intraDppLabel_pre, clssLabel_all = [], [], [], [], [], []
    mini_batch = random.sample(self._memory, 3)
    for i in range(len(mini_batch)):
      idn_conv_feat_val, cls_prob_val, dppLabel, input_clss, intraDppLabel, clssLabel = deepcopy(mini_batch[i])
      feat_cropped_val_all.append(feat_cropped_val)
      pIOU_all.append(pIOU)
      dppLabel_all.append(dppLabel)
      input_clss_all.append(input_clss)
      intraDppLabel_pre.append(intraDppLabel)
      clssLabel_all.append(clssLabel)
    len_input_boxes = np.cumsum(np.insert([len(feat_cropped_val_all[j]) for j in range(len(feat_cropped_val_all))], 0, 0))[:-1]
    feat_cropped_val_all = np.concatenate(feat_cropped_val_all, axis=0)
    input_clss_all = np.concatenate(input_clss_all)
    pIOU_all = scipy.linalg.block_diag(*pIOU_all)
    dppLabel_all = np.concatenate([dppLabel_all[i] + len_input_boxes[i] for i in range(len(dppLabel_all))])

    # intraDppLabel_all, intraDppClss_all = self._idn_intra_label(input_clss_all, dppLabel_all)

    # intraDppLabel_pre = np.concatenate([list(intraDppLabel_pre + len_input_boxes)[i].reshape(-1) for i in range(len(intraDppLabel_pre))])
    # intraDppLabel_pre = intraDppLabel_pre[intraDppLabel_pre>=0]
    # intraDppLabel_clss = input_clss_all[intraDppLabel_pre]
    #
    # max_len = Counter(list(intraDppLabel_clss.squeeze())).most_common(1)[0][1]
    # intraDppLabel_all = np.ones((len(np.unique(intraDppLabel_clss)), max_len)) * (-1)
    # for cnt, ii in enumerate(np.unique(intraDppLabel_clss)):
    #   intraDppLabel_all[cnt, :np.sum(intraDppLabel_clss==ii)] = np.array(intraDppLabel_pre[np.where((intraDppLabel_clss==ii))[0]])
    #
    # clssLabel_all = np.concatenate(list([clssLabel_all[ii] + len_input_boxes[ii] for ii in range(len(len_input_boxes))]))

    while len(feat_cropped_val_all) > 300:
      count = Counter(list(input_clss_all.squeeze()))
      argmax_index = count.most_common(1)[0][0]
      pop_candidate = np.where(input_clss_all.squeeze() == argmax_index)[0]
      in_dpplabel = True
      while in_dpplabel:
        pop_index = random.sample(pop_candidate, 5)
        in_dpplabel = any([pop_index[i] in dppLabel_all for i in range(len(pop_index))])
      feat_cropped_val_all = np.delete(feat_cropped_val_all, pop_index, 0)
      input_clss_all = np.delete(input_clss_all, pop_index, 0)
      for aa in sorted(pop_index, reverse=True):
        dppLabel_all[dppLabel_all > aa] -= 1
      pIOU_all = np.delete(np.delete(pIOU_all, pop_index, 0), pop_index, 1)

    intraDppLabel_all, intraDppClss_all = self._idn_intra_label(input_clss_all, dppLabel_all)

    # print(feat_cropped_val_all.shape[0])
    # print(dppLabel_all.shape)

    return {
      self._rin_placeholder["feat_cropped"]: feat_cropped_val_all,
      self._rin_placeholder["pIOU"]: pIOU_all,
      self._rin_placeholder["dppLabel"]: dppLabel_all,
      self._rin_placeholder["intraDppLabel"]: intraDppLabel_all,
      self._rin_placeholder["clssLabel"]: intraDppClss_all
    }

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    if self.similarity_training:
      sim_train = sess.run(self._idn_sim_values['idn_train'], feed_dict=feed_dict)
      qual_train = False
    elif self.quality_training:
      qual_train = sess.run(self._idn_qual_values['idn_train'], feed_dict=feed_dict)
      sim_train = False
    else:
      qual_train = False
      sim_train = False

    blob = {}
    if self.frcnn_training:
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['total_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        blob['rpn_loss_cls']= rpn_loss_cls
        blob['rpn_loss_box']= rpn_loss_box
        blob['loss_cls']= loss_cls
        blob['loss_box']= loss_box
        blob['loss']= loss
        blob['summary']= summary

    elif self.quality_training:
        if qual_train:
              rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,SS_loss, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                                          self._losses['rpn_loss_box'],
                                                                                                          self._losses['cross_entropy'],
                                                                                                          self._losses['loss_box'],
                                                                                                          self._losses["SS_loss"],
                                                                                                          self._losses['total_loss'],
                                                                                                          self._summary_op,
                                                                                                          train_op],
                                                                                                         feed_dict=feed_dict)
              blob['rpn_loss_cls']= rpn_loss_cls
              blob['rpn_loss_box']= rpn_loss_box
              blob['loss_cls']= loss_cls
              blob['loss_box']= loss_box
              blob['SS_loss']= SS_loss
              blob['loss']= loss
              blob['summary']= summary
        else:
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, SS_loss, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                                self._losses['rpn_loss_box'],
                                                                                                self._losses['cross_entropy'],
                                                                                                self._losses['loss_box'],
                                                                                                self._losses["SS_loss"],
                                                                                                self._losses['total_loss'],
                                                                                                train_op],
                                                                                               feed_dict=feed_dict)
            blob['rpn_loss_cls'] = rpn_loss_cls
            blob['rpn_loss_box'] = rpn_loss_box
            blob['loss_cls'] = loss_cls
            blob['loss_box'] = loss_box
            blob['SS_loss'] = SS_loss
            blob['loss'] = loss

    elif self.similarity_training:
        if sim_train:
            ID_loss, summary, _ = sess.run([self._losses["ID_loss"],
                                             self._summary_op,
                                              train_op],
                                  feed_dict=feed_dict)
            blob['ID_loss']= ID_loss
            blob['summary']= summary
        else:
            ID_loss, _ = sess.run([self._losses["ID_loss"],
                                              train_op],
                                  feed_dict=feed_dict)
            blob['ID_loss']= ID_loss

    return blob, sim_train, qual_train


  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)

