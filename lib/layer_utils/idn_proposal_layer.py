import numpy as np
from model.bbox_transform import bbox_transform_inv
from utils.cython_bbox import bbox_overlaps
# from utils.cython_dpp_bbox import dpp_bbox_overlaps
from utils.boxTools import *
from utils.myutils import *
from scipy.optimize import linear_sum_assignment
from model.config import cfg
import copy

def idn_qual_proposal_layer(scores, box_deltas, gts, rois, im_info, num_clss):
    input_boxes, input_scores, input_clss, gt_overlaps, pIOU, assign_gt_ind, dppLabel, mask, add_gt, posDppLabel = \
        _roi_preprocessing(scores, box_deltas, gts, rois, im_info, num_clss, thresh=0.7, top_n=cfg.TRAIN.QUAL_TOPN, scores_thresh=0.001)

    input_boxes = np.column_stack((np.zeros(np.shape(input_boxes)[0]), input_boxes))
    negDppLabel = np.where((np.max(gt_overlaps, 0) < 0.5))[0]
    clss = np.reshape(np.eye(num_clss)[np.squeeze(input_clss)], [-1, num_clss])
    num_patch = np.array([1])
    if len(input_boxes)==0 or len(dppLabel)>15:
        input_boxes = np.array([[0, 0, 0, 1, 1]]).astype(np.float32, copy=False)
        input_clss = np.zeros((1, 1))
        input_scores = np.zeros((1,))
        pIOU = np.zeros((1, 1))
        num_patch = np.array([0])
        clss = np.zeros((1, num_clss))

    return input_boxes.astype(np.float32, copy=False), input_clss.astype(np.int32, copy=False), \
           input_scores.astype(np.float32, copy=False), pIOU.astype(np.float32, copy=False), \
           clss.astype(np.float32, copy=False), assign_gt_ind.astype(np.int32, copy=False), \
           num_patch.astype(np.int32, copy=False), dppLabel.astype(np.int32, copy=False), \
           mask.astype(np.int32, copy=False), add_gt, negDppLabel.astype(np.int32, copy=False), \
           posDppLabel.astype(np.int32, copy=False)

def idn_sim_proposal_layer(scores, box_deltas, gts, rois, im_info, num_clss):
    input_boxes, input_scores, input_clss, gt_overlaps, pIOU, assign_gt_ind, dppLabel, mask, add_gt, _ = \
        _preprocessing(scores, box_deltas, gts, rois, im_info, num_clss, thresh=0.7, top_n=(num_clss-1), scores_thresh=0.001)
    intraDppLabel, clssLabel = _idn_intra_label(input_clss, dppLabel)

    input_boxes = np.column_stack((np.zeros(np.shape(input_boxes)[0]), input_boxes))
    clss = np.reshape(np.eye(num_clss)[np.squeeze(input_clss)], [-1, num_clss])
    num_patch = np.array([1])
    if len(input_boxes)==0 or len(dppLabel)>15:
        input_boxes = np.array([[0, 0, 0, 1, 1]]).astype(np.float32, copy=False)
        input_clss = np.zeros((1, 1))
        input_scores = np.zeros((1,))
        pIOU = np.zeros((1, 1))
        num_patch = np.array([0])
        clss = np.zeros((1, num_clss))

    return input_boxes.astype(np.float32, copy=False), input_clss.astype(np.int32, copy=False), \
           input_scores.astype(np.float32, copy=False), pIOU.astype(np.float32, copy=False), \
           clss.astype(np.float32, copy=False), assign_gt_ind.astype(np.int32, copy=False), \
           num_patch.astype(np.int32, copy=False), dppLabel.astype(np.int32, copy=False), \
           intraDppLabel.astype(np.int32, copy=False), clssLabel.astype(np.int32, copy=False), \
           mask.astype(np.int32, copy=False), add_gt, \

def idn_sim_target_layer(gts, im_info, num_clss):
    input_boxes, input_clss, gt_overlaps, pIOU, assign_gt_ind, dppLabel = \
        _gt_preprocessing(gts, im_info)
    intraDppLabel, clssLabel = _idn_intra_label(input_clss, dppLabel)

    input_boxes = np.column_stack((np.zeros(np.shape(input_boxes)[0]), input_boxes))
    clss = np.reshape(np.eye(num_clss)[np.squeeze(input_clss)], [-1, num_clss])
    num_patch = np.array([1])
    if len(input_boxes)==0 or len(dppLabel)>15:
        input_boxes = np.array([[0, 0, 0, 1, 1]]).astype(np.float32, copy=False)
        input_clss = np.zeros((1, 1))
        pIOU = np.zeros((1, 1))
        num_patch = np.array([0])
        clss = np.zeros((1, num_clss))

    return input_boxes.astype(np.float32, copy=False), input_clss.astype(np.int32, copy=False), \
           pIOU.astype(np.float32, copy=False), \
           clss.astype(np.float32, copy=False), assign_gt_ind.astype(np.int32, copy=False), \
           num_patch.astype(np.int32, copy=False), dppLabel.astype(np.int32, copy=False), \
           intraDppLabel.astype(np.int32, copy=False), clssLabel.astype(np.int32, copy=False)

def _roi_preprocessing(scores, box_deltas, gts, rois, im_info, num_clss, thresh, top_n, scores_thresh):
    add_gt = False
    mask = np.zeros(scores.shape)
    boxes = rois[:, 1:5]
    all_boxes = bbox_transform_inv(boxes, box_deltas)
    all_boxes = _clip_boxes(all_boxes, im_info[:2])
    gt_boxes = gts[:, :4]
    gt_clss = gts[:, 4].astype(np.int32)
    scores = scores[:, 1:]
    scores_new = copy.deepcopy(scores)
    M0 = all_boxes.shape[0]
    num_ignored = scores_new.shape[1] - 1
    sorted_scores = np.argsort(-scores_new, 1)
    ignored_cols = np.reshape(sorted_scores[:, -num_ignored:], (M0 * num_ignored))
    ignored_rows = np.repeat(range(0, sorted_scores.shape[0]), num_ignored)
    scores_new[ignored_rows, ignored_cols] = 0.0
    top1_index = np.nonzero(scores_new >= scores_thresh)
    top1_index = np.transpose(np.stack(top1_index))

    scores_new = copy.deepcopy(scores)
    num_ignored = scores_new.shape[1] - top_n
    sorted_scores = np.argsort(-scores_new, 1)
    ignored_cols = np.reshape(sorted_scores[:, -num_ignored:], (M0 * num_ignored))
    ignored_rows = np.repeat(range(0, sorted_scores.shape[0]), num_ignored)
    scores_new[ignored_rows, ignored_cols] = 0.0
    top_n_index = high_scores= np.nonzero(scores_new >= scores_thresh)
    top_n_index = np.transpose(np.stack(top_n_index))
    pos_label = np.where(inNd(top_n_index, top1_index))[0]

    lbl_high_scores = high_scores[1]
    box_high_scores = high_scores[0]
    input_scores = np.reshape(scores[box_high_scores, lbl_high_scores], (lbl_high_scores.shape[0],))
    input_boxes = np.reshape(
        all_boxes[np.tile(box_high_scores, 4), np.hstack((np.multiply(4,lbl_high_scores), np.add(np.multiply(4,lbl_high_scores),1), \
                                                          np.add(np.multiply(4, lbl_high_scores), 2),
                                                          np.add(np.multiply(4, lbl_high_scores), 3)))],
        (lbl_high_scores.shape[0], 4), order='F')
    input_clss = np.add(lbl_high_scores,1)

    index_row = np.where((scores_new >= scores_thresh))[0]
    index_column = np.where((scores_new >= scores_thresh))[1]
    mask[:, 1:][index_row, index_column] = 1

    if len(input_boxes) < 1:
        input_boxes = np.vstack((input_boxes, gt_boxes))
        input_scores = np.hstack((input_scores, [1] * len(gt_boxes)))
        input_clss = np.hstack((input_clss, gt_clss))
        add_gt = True
    else:
        gt_overlaps_temp = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                                 np.ascontiguousarray(input_boxes, dtype=np.float))
        assign_gt_ind_temp = np.argmax(gt_overlaps_temp, 0)
        assign_gt_ind_temp = np.where((np.max(gt_overlaps_temp, 0) > 0.8) & (gt_clss[assign_gt_ind_temp] == input_clss))[0]
        if len(assign_gt_ind_temp)<1:
            input_boxes = np.vstack((input_boxes, gt_boxes))
            input_scores = np.hstack((input_scores, [1] * len(gt_boxes)))
            input_clss = np.hstack((input_clss, gt_clss))
            add_gt = True
        else:
            gt_overlaps_temp = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                                     np.ascontiguousarray(input_boxes[assign_gt_ind_temp], dtype=np.float))
            gt_overlaps_temp = 1/gt_overlaps_temp
            gt_overlaps_temp[np.isinf(gt_overlaps_temp)] = 100000
            if len(linear_sum_assignment(gt_overlaps_temp)[1]) != len(gt_clss):
                input_boxes = np.vstack((input_boxes, gt_boxes))
                input_scores = np.hstack((input_scores, [1] * len(gt_boxes)))
                input_clss = np.hstack((input_clss, gt_clss))
                add_gt = True

    gt_overlaps = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                             np.ascontiguousarray(input_boxes, dtype=np.float))
    pIOU = bbox_overlaps(np.ascontiguousarray(input_boxes, dtype=np.float),
                         np.ascontiguousarray(input_boxes, dtype=np.float))
    assign_gt_ind = np.argmax(gt_overlaps, 0)
    gt_overlaps_temp = 1/gt_overlaps
    gt_overlaps_temp[np.isinf(gt_overlaps_temp)] = 100000
    dppLabel = linear_sum_assignment(gt_overlaps_temp)[1]

    return input_boxes, input_scores, input_clss, gt_overlaps, pIOU, assign_gt_ind, dppLabel, mask, add_gt, pos_label

def _preprocessing(scores, box_deltas, gts, rois, im_info, num_clss, thresh, top_n, scores_thresh):
    add_gt = False
    mask = np.zeros(scores.shape)
    boxes = rois[:, 1:5]
    all_boxes = bbox_transform_inv(boxes, box_deltas)
    all_boxes = _clip_boxes(all_boxes, im_info[:2])
    gt_boxes = gts[:, :4]
    gt_clss = gts[:, 4].astype(np.int32)
    scores = scores[:, 1:]
    if top_n != (num_clss-1):
        M0 = all_boxes.shape[0]
        num_ignored = scores.shape[1] - top_n
        sorted_scores = np.argsort(-scores, 1)
        ignored_cols = np.reshape(sorted_scores[:, -num_ignored:], (M0 * num_ignored))
        ignored_rows = np.repeat(range(0, sorted_scores.shape[0]), num_ignored)
        scores[ignored_rows, ignored_cols] = 0
    high_scores = np.nonzero(scores >= scores_thresh)
    lbl_high_scores = high_scores[1]
    box_high_scores = high_scores[0]
    input_scores = np.reshape(scores[box_high_scores, lbl_high_scores], (lbl_high_scores.shape[0],))
    input_boxes = np.reshape(
        all_boxes[np.tile(box_high_scores, 4), np.hstack((np.multiply(4,lbl_high_scores), np.add(np.multiply(4,lbl_high_scores),1), \
                                                          np.add(np.multiply(4, lbl_high_scores), 2),
                                                          np.add(np.multiply(4, lbl_high_scores), 3)))],
        (lbl_high_scores.shape[0], 4), order='F')
    input_clss = np.add(lbl_high_scores,1)
    # Thresholding
    overlaps_temp = bbox_overlaps(
        np.ascontiguousarray(gt_boxes, dtype=np.float),
        np.ascontiguousarray(input_boxes, dtype=np.float))
    index = np.where(np.max(overlaps_temp, 0) > thresh)[0]
    input_scores = input_scores[index]
    input_boxes = input_boxes[index]
    input_clss = input_clss[index]

    if len(input_boxes) < 1:
        input_boxes = np.vstack((input_boxes, gt_boxes))
        input_scores = np.hstack((input_scores, [1] * len(gt_boxes)))
        input_clss = np.hstack((input_clss, gt_clss))
        add_gt = True
    else:
        gt_overlaps_temp = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                                 np.ascontiguousarray(input_boxes, dtype=np.float))
        assign_gt_ind_temp = np.argmax(gt_overlaps_temp, 0)
        assign_gt_ind_temp = np.where((np.max(gt_overlaps_temp, 0) > 0.8) & (gt_clss[assign_gt_ind_temp] == input_clss))[0]
        if len(assign_gt_ind_temp)<1:
            input_boxes = np.vstack((input_boxes, gt_boxes))
            input_scores = np.hstack((input_scores, [1] * len(gt_boxes)))
            input_clss = np.hstack((input_clss, gt_clss))
            add_gt = True
        else:
            gt_overlaps_temp = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                                     np.ascontiguousarray(input_boxes[assign_gt_ind_temp], dtype=np.float))
            gt_overlaps_temp = 1/gt_overlaps_temp
            gt_overlaps_temp[np.isinf(gt_overlaps_temp)] = 100000
            if len(linear_sum_assignment(gt_overlaps_temp)[1]) != len(gt_clss):
                input_boxes = np.vstack((input_boxes, gt_boxes))
                input_scores = np.hstack((input_scores, [1] * len(gt_boxes)))
                input_clss = np.hstack((input_clss, gt_clss))
                add_gt = True

    index_row = np.where((scores >= scores_thresh))[0][index]
    index_column = np.where((scores >= scores_thresh))[1][index]
    mask[:, 1:][index_row, index_column] = 1

    gt_overlaps = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                             np.ascontiguousarray(input_boxes, dtype=np.float))
    pIOU = bbox_overlaps(np.ascontiguousarray(input_boxes, dtype=np.float),
                         np.ascontiguousarray(input_boxes, dtype=np.float))
    assign_gt_ind = np.argmax(gt_overlaps, 0)
    gt_overlaps_temp = 1/gt_overlaps
    gt_overlaps_temp[np.isinf(gt_overlaps_temp)] = 100000
    dppLabel = linear_sum_assignment(gt_overlaps_temp)[1]
    posDppLabel = np.where((np.max(gt_overlaps, 0) > 0.5))[0]

    return input_boxes, input_scores, input_clss, gt_overlaps, pIOU, assign_gt_ind, dppLabel, mask, add_gt, posDppLabel

def _gt_preprocessing(gts, im_info):
    gt_boxes = gts[:, :4]
    gt_clss = gts[:, 4].astype(np.int32)
    num_noise = int(cfg.TRAIN.NUM_SIM_TARGET_BOXES/len(gts))

    noise = []
    for im_idx in range(len(gt_boxes)):
        x_sigma = (gt_boxes[im_idx][2] - gt_boxes[im_idx][0]) / cfg.TRAIN.SIM_VAR
        y_sigma = (gt_boxes[im_idx][3] - gt_boxes[im_idx][1]) / cfg.TRAIN.SIM_VAR
        x1 = np.random.normal(0, x_sigma, num_noise)
        x1 = x1 - max(x1)
        y1 = np.random.normal(0, y_sigma, num_noise)
        y1 = y1 - max(y1)
        x2 = np.random.normal(0, x_sigma, num_noise)
        x2 = x2 - min(x2)
        y2 = np.random.normal(0, y_sigma, num_noise)
        y2 = y2 - min(y2)
        noise.append(np.concatenate([x1, y1, x2, y2]))

    noise = np.reshape(np.stack(noise), [num_noise * len(gt_boxes), 4])

    input_boxes = np.repeat(gt_boxes, num_noise, axis=0) + noise
    input_clss = np.repeat(gt_clss, num_noise)

    input_boxes = np.vstack((input_boxes, gt_boxes))
    input_clss = np.hstack((input_clss, gt_clss))

    order = np.arange(len(input_boxes))
    np.random.shuffle(order)
    input_boxes = input_boxes[order]
    input_clss = input_clss[order]
    inside = np.where((input_boxes[:,0] >= 0.) & (input_boxes[:,1] >= 0.) & (input_boxes[:,2] <= im_info[1]) & (input_boxes[:,3] <= im_info[0]))[0]
    input_boxes = input_boxes[inside]
    input_clss = input_clss[inside]

    gt_overlaps = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                             np.ascontiguousarray(input_boxes, dtype=np.float))
    pIOU = bbox_overlaps(np.ascontiguousarray(input_boxes, dtype=np.float),
                         np.ascontiguousarray(input_boxes, dtype=np.float))

    assign_gt_ind = np.argmax(gt_overlaps, 0)
    gt_overlaps_temp = 1/gt_overlaps
    gt_overlaps_temp[np.isinf(gt_overlaps_temp)] = 100000
    dppLabel = linear_sum_assignment(gt_overlaps_temp)[1]

    return input_boxes, input_clss, gt_overlaps, pIOU, assign_gt_ind, dppLabel

def _idn_label(input_boxes, gts):
    gt_boxes = gts[:,:4]
    gt_overlaps = bbox_overlaps(np.ascontiguousarray(gt_boxes, dtype=np.float),
                             np.ascontiguousarray(input_boxes, dtype=np.float))

    return np.argmax(gt_overlaps, 1)

def _idn_intra_label(input_clss, dppLabel):
    intraDppClss = [np.where(input_clss == cls)[0] for cls in np.unique(input_clss) if
                    len(np.where(input_clss == cls)[0]) > 0]
    intraDppLabel = [np.array(list(set(dppLabel).intersection(intraDppClss[i])))
                     for i in range(len(intraDppClss))]

    intraDppClss = [intraDppClss[i] for i in range(len(intraDppLabel)) if len(intraDppLabel[i]) > 0]
    intraDppLabel = [intraDppLabel[i] for i in range(len(intraDppLabel)) if len(intraDppLabel[i]) > 0]

    intraDppClss_pre = [intraDppClss[i] for i in range(len(intraDppLabel)) if len(intraDppLabel[i]) > 0]
    max_len = np.max([len(intraDppClss_pre[i]) for i in range(len(intraDppClss_pre))])
    intraDppClss = np.ones((len(intraDppClss_pre), max_len)) * (-1)
    for ii in range(len(intraDppClss)):
        intraDppClss[ii, :len(intraDppClss_pre[ii])] = np.array(intraDppClss_pre[ii])

    intraDppLabel_pre = [intraDppLabel[i] for i in range(len(intraDppLabel)) if len(intraDppLabel[i]) > 0]
    max_len = np.max([len(intraDppLabel_pre[i]) for i in range(len(intraDppLabel_pre))])
    intraDppLabel = np.ones((len(intraDppLabel_pre), max_len)) * (-1)
    for ii in range(len(intraDppLabel_pre)):
        intraDppLabel[ii, :len(intraDppLabel_pre[ii])] = np.array(intraDppLabel_pre[ii])

    return intraDppLabel, intraDppClss

def idn_proposal_test_layer(scores, box_deltas, rois, im_info, num_clss):
    thresh = 0.0001
    mask = np.zeros(scores.shape)
    boxes = rois[:, 1:5]
    all_boxes = bbox_transform_inv(boxes, box_deltas)
    all_boxes = _clip_boxes(all_boxes, im_info[:2])
    scores = scores[:, 1:]

    high_scores = np.nonzero(scores >= thresh)
    lbl_high_scores = high_scores[1]
    box_high_scores = high_scores[0]
    input_scores = np.reshape(scores[box_high_scores, lbl_high_scores], (lbl_high_scores.shape[0],))
    all_boxes = all_boxes[:, 4:]
    input_boxes = np.reshape(
        all_boxes[np.tile(box_high_scores, 4), np.hstack((np.multiply(4,lbl_high_scores), np.add(np.multiply(4,lbl_high_scores),1), \
                                                          np.add(np.multiply(4, lbl_high_scores), 2),
                                                          np.add(np.multiply(4, lbl_high_scores), 3)))],
        (lbl_high_scores.shape[0], 4), order='F')
    input_clss = np.add(lbl_high_scores,1)

    index_row = np.where((scores >= thresh))[0]
    index_column = np.where((scores >= thresh))[1]
    mask[:, 1:][index_row, index_column] = 1

    pIOU = bbox_overlaps(np.ascontiguousarray(input_boxes, dtype=np.float),
                         np.ascontiguousarray(input_boxes, dtype=np.float))

    input_boxes = np.column_stack((np.zeros(np.shape(input_boxes)[0]), input_boxes))
    clss = np.reshape(np.eye(num_clss)[np.squeeze(input_clss)], [-1, num_clss])
    num_patch = np.array([1])
    if len(input_boxes)==0:
        input_boxes = np.array([[0, 0, 0, 1, 1]]).astype(np.float32, copy=False)
        input_clss = np.zeros((1, 1))
        input_scores = np.zeros((1,))
        pIOU = np.zeros((1, 1))
        num_patch = np.array([0])
        clss = np.zeros((1, num_clss))

    return input_boxes.astype(np.float32, copy=False), input_clss.astype(np.int32, copy=False), \
           input_scores.astype(np.float32, copy=False),\
           pIOU.astype(np.float32, copy=False), num_patch.astype(np.int32, copy=False), \
           clss.astype(np.float32, copy=False), mask.astype(np.int32, copy=False)

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

