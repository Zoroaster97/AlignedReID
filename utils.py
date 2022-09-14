import cv2, torch
import numpy as np

def normalize(x, axis=-1, scale=1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x * scale


def EuclideanDistances(a, b):
    sq_a = a**2
    # sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sum_sq_a = torch.sum(sq_a, dim=1)
    sq_b = b**2
    # sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    sum_sq_b = torch.sum(sq_b, dim=1)
    bt = b.t()
    return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))


def get_acc_prob_hist(hist):
    acc_hist = np.zeros([256, 1], np.float32)
    pre_val = 0
    for i in range(256):
        acc_hist[i, 0] = pre_val + hist[i, 0]
        pre_val = acc_hist[i, 0]
    acc_hist /= pre_val
    return acc_hist

def get_hist_color_hsv(img):
    # h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    img_hist = cv2.calcHist([v], [0], None, [256], [0.0, 255.0])
    img_acc_prob_hist = get_acc_prob_hist(img_hist)
    return img_acc_prob_hist

def specify_hist_color_hsv(src_acc_prob_hist, dst_img):
    # dst_H, dst_S, dst_V = cv2.split(cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV))
    dst_H, dst_S, dst_V = cv2.split(cv2.cvtColor(dst_img, cv2.COLOR_RGB2HSV))
    dst_hist = cv2.calcHist([dst_V], [0], None, [256], [0.0, 255.0])
    dst_acc_prob_hist = get_acc_prob_hist(dst_hist)
    diff_acc_prob = abs(np.tile(src_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_acc_prob_hist.reshape(1, 256))
    table = np.argmin(diff_acc_prob, axis=0).astype(np.int8)
    sp_V = cv2.LUT(dst_V, table).astype(np.uint8)
    # sp_image = cv2.cvtColor(cv2.merge([dst_H, dst_S, sp_V]), cv2.COLOR_HSV2BGR)
    sp_image = cv2.cvtColor(cv2.merge([dst_H, dst_S, sp_V]), cv2.COLOR_HSV2RGB)
    return sp_image