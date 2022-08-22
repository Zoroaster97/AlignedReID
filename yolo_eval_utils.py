import os, cv2, torch
# from pytorchyolo import detect, models
# from aligned_reid.model.Model import Model
# import torchvision.transforms as T
import numpy as np
# import time

class Evaluater:
    def __init__(self, dirname, anno_filename='yolo_anno.txt', iou_threshold=0.9):
        self.anno_file_name = os.path.join(dirname, anno_filename)
        self.annos = []
        self.str_annos = []
        # print(self.anno_file_name)
        assert os.path.exists(self.anno_file_name)
        self.load_annos()
        self.test_num = 0   # 检测到超过一个人的图像数量
        self.fail_count = 0 # 匹配失败的次数
        self.succ_count = 0 # 匹配成功的次数
        self.iou_threshold = iou_threshold
    
    def load_annos(self):
        with open(self.anno_file_name) as f:
            self.str_annos = f.readlines()
        self.annos = []
        for str_anno in self.str_annos:
            x, y, w, h = str_anno.strip().split(',')
            self.annos.append([int(x), int(y), int(w), int(h)])
    
    def eval(self, idx, box):
        assert self.fail_count + self.succ_count == self.test_num
        anno_box = self.annos[idx]
        overlap_x1 = max(box[0], anno_box[0])
        overlap_y1 = max(box[1], anno_box[1])
        overlap_x2 = min(box[0] + box[2], anno_box[0] + anno_box[2])
        overlap_y2 = min(box[1] + box[3], anno_box[1] + anno_box[3])
        intsc = max(overlap_x2 - overlap_x1, 0) * max(overlap_y2 - overlap_y1, 0)
        sbox = box[2] * box[3]
        sannobox = anno_box[2] * anno_box[3]
        union = sbox + sannobox - intsc
        iou = intsc / union
        if iou >= self.iou_threshold:
            self.succ_count += 1
        else:
            self.fail_count += 1
            # print('failed:', idx)
        self.test_num += 1
    
    def get_eval_result(self):
        assert self.fail_count + self.succ_count == self.test_num
        result = {}
        result['test_num'] = self.test_num
        result['fail_count'] = self.fail_count
        result['succ_count'] = self.succ_count
        result['fail_rate'] = self.fail_count / self.test_num
        result['succ_rate'] = self.succ_count / self.test_num
        return result


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def EuclideanDistances(a,b):
    sq_a = a**2
    # sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sum_sq_a = torch.sum(sq_a,dim=1)
    sq_b = b**2
    # sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    sum_sq_b = torch.sum(sq_b,dim=1)
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))


def get_acc_prob_hist(hist):
    acc_hist = np.zeros([256, 1], np.float32)
    pre_val = 0
    for i in range(256):
        acc_hist[i, 0] = pre_val + hist[i, 0]
        pre_val = acc_hist[i, 0]
    acc_hist /= pre_val
    return acc_hist

def get_hist_color_hsv(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    img_hist = cv2.calcHist([v], [0], None, [256], [0.0, 255.0])
    img_acc_prob_hist = get_acc_prob_hist(img_hist)
    return img_acc_prob_hist

def specify_hist_color_hsv(src_acc_prob_hist, dst_img):
    dst_H, dst_S, dst_V = cv2.split(cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV))
    dst_hist = cv2.calcHist([dst_V], [0], None, [256], [0.0, 255.0])
    dst_acc_prob_hist = get_acc_prob_hist(dst_hist)
    diff_acc_prob = abs(np.tile(src_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_acc_prob_hist.reshape(1, 256))
    table = np.argmin(diff_acc_prob, axis=0).astype(np.int8)
    sp_V = cv2.LUT(dst_V, table).astype(np.uint8)
    sp_image = cv2.cvtColor(cv2.merge([dst_H, dst_S, sp_V]), cv2.COLOR_HSV2BGR)
    return sp_image

def intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def center_error(rects1, rects2):
    r"""Center error.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))

    return errors


def normalized_center_error(rects1, rects2):
    r"""Center error normalized by the size of ground truth.

    Args:
        rects1 (numpy.ndarray): prediction box. An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): groudn truth box. An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(
        np.sum(np.power((centers1 - centers2) /
                        np.maximum(np.array([[1., 1.]]), rects2[:, 2:]), 2),
               axis=-1))

    return errors


def calc_metrics(boxes, anno):
    ious = rect_iou(boxes, anno)
    center_errors = center_error(boxes, anno)
    norm_center_errors = normalized_center_error(
        boxes, anno)
    return ious, center_errors, norm_center_errors


NBINS_IOU = 21
NBINS_CE = 51
NBINS_NCE = 51

def calc_curves(ious, center_errors, norm_center_errors):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]
    norm_center_errors = np.asarray(norm_center_errors,
                                    float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, NBINS_IOU)[np.newaxis, :]
    thr_ce = np.arange(0, NBINS_CE)[np.newaxis, :]
    thr_nce = np.linspace(0, 0.5, NBINS_NCE)[np.newaxis, :]

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)
    bin_nce = np.less_equal(norm_center_errors, thr_nce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)
    norm_prec_curve = np.mean(bin_nce, axis=0)

    return succ_curve, prec_curve, norm_prec_curve

