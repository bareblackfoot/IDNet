import numpy as np

from model.config import cfg

from sklearn import preprocessing
from utils.timer import Timer
import time
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import operator
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def scatter(x, colors, sorted_feat, sorted_obj, cnt, im_idx):
    num_objects = np.max(np.unique(colors)) + 1
    palette = np.array(sns.color_palette("hls", num_objects))

    # We create a self.scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])

    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in np.unique(colors):
        # Position of each label.
        xtext, ytext = np.median(x[np.where(colors == i)[0], :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=14, color=palette[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    f.savefig("/home/blackfoot/git/tf-faster-rcnn/output/vgg16/metric/tsne_%d_%d.png" % (im_idx, cnt))
    plt.close(f)

    g = plt.figure()
    ax = plt.subplot(aspect='equal')
    ax.grid(False)
    sorted_feat = preprocessing.normalize(sorted_feat, norm='l2')
    mat = ax.imshow(np.matmul(sorted_feat, sorted_feat.T), cmap='gray')
    ax.set_xticklabels(np.insert(sorted_obj, 0, 0))
    ax.set_yticklabels(np.insert(sorted_obj, 0, 0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    g.savefig("/home/blackfoot/git/tf-faster-rcnn/output/vgg16/metric/matrix_%d_%d.png" % (im_idx, cnt))
    plt.close(g)

def load(data_path, session, saver, name=None, ignore_missing=False):
    if data_path.endswith('.npy'):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(name, reuse=True):
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign " + name + '/' + key + '/' + subkey
                        except ValueError:
                            print "ignore " + name + '/' + key + '/' + subkey
                            if not ignore_missing:
                                raise
    else:
        reader = pywrap_tensorflow.NewCheckpointReader(data_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            with tf.variable_scope("", reuse=True):
                try:
                    var = tf.get_variable(key)
                    session.run(var.assign(reader.get_tensor(key)))
                    print "assign pretrain model " + key
                except ValueError:
                    print "ignore " + key

def inNd(a, b, assume_unique=False):
    a = np.asarray(a, order='C')
    b = np.asarray(b, order='C')
    a = a.ravel().view((np.str, a.itemsize * a.shape[1]))
    b = b.ravel().view((np.str, b.itemsize * b.shape[1]))
    return np.in1d(a, b, assume_unique)

def dpp_infer(S, q, pIOU, sim_thresh, thresh):
    q = np.reshape(q, [-1, 1])
    S_sum = np.add(np.multiply(0.6, S), np.multiply(0.4, pIOU)) + cfg.EPSILON*np.identity(len(q))
    L = np.sqrt(np.tile(q, [1, len(S)])) * S_sum * np.sqrt(np.tile(q.T, [len(S), 1]))
    L = np.divide(L + np.transpose(L), 2.)
    selected = [np.argmax(q)]
    oldS = S_sum[selected, :][:, selected]
    remains = list(range(len(q)))
    picks = []
    remains.pop(selected[0])
    picks.append(selected[0])
    oldProb = np.log(np.linalg.det(oldS)) + np.log(reduce(lambda x, y: np.multiply(x,y),q[picks]))
    maxProb = oldProb
    while len(remains)>0:
        tempProb = -100*np.ones_like(range(len(q)),dtype=np.float32)
        A = L[picks, :][:, picks]
        detA = np.linalg.det(A)
        C = L[picks, :][:, remains]
        b = np.diag(L[remains,:][:,remains])
        tempProb[remains] = np.log(np.multiply(b - np.diag(np.matmul(np.matmul(C.T, np.linalg.inv(A)), C)), detA))
        tempProb[np.isnan(tempProb)] = -100
        tempProb[np.isinf(tempProb)] = -100
        candi = np.where(tempProb - maxProb > thresh)[0]
        candi_sorted = candi[np.argsort(-tempProb[candi])]
        cond_candi = np.where((np.max(S_sum[candi_sorted, :][:, picks], 1) < sim_thresh))[0]
        if len(cond_candi) > 0:
            selected = candi_sorted[cond_candi[0]]
            picks.append(selected)
            remains.pop(remains.index(selected))
            maxProb = tempProb[selected]
        else:
            return picks
    return picks

def dpp_dh(pred_si, pred_s, num_proposal,sIOU, alpha):
    L_si = alpha*np.matmul(pred_si,pred_si.T) + (1-alpha)*sIOU
    L_pred = np.sqrt(np.tile(np.expand_dims(pred_s,-1),[1, num_proposal]))*L_si*np.sqrt(np.tile(np.transpose(np.expand_dims(pred_s,-1)),[num_proposal, 1]))
    selected = [np.argmax(pred_s)]
    oldL = L_pred[selected, :][:, selected]
    remains = list(range(num_proposal))
    picks = []
    remains.pop(selected[0])
    picks.append(selected[0])
    oldDet = np.linalg.det(oldL)
    maxDet = oldDet
    while 1:
        for i in remains:
            orig_len = len(oldL)
            temp = np.zeros((orig_len + 1, orig_len + 1))
            temp[:orig_len, :orig_len] = oldL
            new_feat = L_pred[picks, i]
            temp[orig_len, :-1] = new_feat.T
            temp[:-1, orig_len] = new_feat
            temp[-1, -1] = L_pred[i, i]

            if np.linalg.det(temp) > maxDet:
                selected = i
                maxDet = np.linalg.det(temp)
                maxL = temp.copy()

        if maxDet > oldDet + 0.0001:
            picks.append(selected)
            remains.pop(np.where([remains[j] == selected for j in range(len(remains))])[0])
            oldDet = maxDet
            oldL = maxL
        else:
            break
    return picks

def classToString(_classes, clss):
    return _classes[clss]

def bbox_intersection(rois1, rois2):
    size1 = np.shape(rois1)[0]
    size2 = np.shape(rois2)[0]
    overlaps_child_x = np.maximum(0, np.minimum(np.repeat(np.reshape(rois1[:, 2], [size1, 1]), size2, axis=1),
                                                (np.repeat(np.reshape(rois2[:, 2], [size2, 1]), size1,
                                                           axis=1)).T) - np.maximum(
        np.repeat(np.reshape(rois1[:, 0], [size1, 1]), size2, axis=1),
        (np.repeat(np.reshape(rois2[:, 0], [size2, 1]), size1, axis=1)).T) + 1)

    overlaps_child_y = np.maximum(0, np.minimum(np.repeat(np.reshape(rois1[:, 3], [size1, 1]), size2, axis=1),
                                                (np.repeat(np.reshape(rois2[:, 3], [size2, 1]), size1,
                                                           axis=1)).T) - np.maximum(
        np.repeat(np.reshape(rois1[:, 1], [size1, 1]), size2, axis=1),
        (np.repeat(np.reshape(rois2[:, 1], [size2, 1]), size1, axis=1)).T) + 1)

    overlaps_child = np.multiply(overlaps_child_x, overlaps_child_y)
    return overlaps_child