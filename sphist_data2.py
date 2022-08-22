import cv2, os, tqdm
import numpy as np


def get_acc_prob_hist(hist):
    acc_hist = np.zeros([256, 1], np.float32)
    pre_val = 0
    for i in range(256):
        acc_hist[i, 0] = pre_val + hist[i, 0]
        pre_val = acc_hist[i, 0]
    acc_hist /= pre_val
    return acc_hist

def specify_hist_color_hsv(src_img, dst_img):
    src_H, src_S, src_V = cv2.split(cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV))
    dst_H, dst_S, dst_V = cv2.split(cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV))
    src_hist = cv2.calcHist([src_V], [0], None, [256], [0.0, 255.0])
    dst_hist = cv2.calcHist([dst_V], [0], None, [256], [0.0, 255.0])
    src_acc_prob_hist = get_acc_prob_hist(src_hist)
    dst_acc_prob_hist = get_acc_prob_hist(dst_hist)
    diff_acc_prob = abs(np.tile(src_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_acc_prob_hist.reshape(1, 256))
    table = np.argmin(diff_acc_prob, axis=0).astype(np.int8)
    sp_V = cv2.LUT(dst_V, table).astype(np.uint8)
    sp_image = cv2.cvtColor(cv2.merge([dst_H, dst_S, sp_V]), cv2.COLOR_HSV2BGR)
    return sp_image

def specify_hist_color_BGR(src_img, dst_img):
    src_B, src_G, src_R = cv2.split(src_img)
    dst_B, dst_G, dst_R = cv2.split(dst_img)
    src_B_hist = cv2.calcHist([src_B], [0], None, [256], [0.0, 255.0])
    dst_B_hist = cv2.calcHist([dst_B], [0], None, [256], [0.0, 255.0])
    src_G_hist = cv2.calcHist([src_G], [0], None, [256], [0.0, 255.0])
    dst_G_hist = cv2.calcHist([dst_G], [0], None, [256], [0.0, 255.0])
    src_R_hist = cv2.calcHist([src_R], [0], None, [256], [0.0, 255.0])
    dst_R_hist = cv2.calcHist([dst_R], [0], None, [256], [0.0, 255.0])
    src_B_acc_prob_hist = get_acc_prob_hist(src_B_hist)
    dst_B_acc_prob_hist = get_acc_prob_hist(dst_B_hist)
    src_G_acc_prob_hist = get_acc_prob_hist(src_G_hist)
    dst_G_acc_prob_hist = get_acc_prob_hist(dst_G_hist)
    src_R_acc_prob_hist = get_acc_prob_hist(src_R_hist)
    dst_R_acc_prob_hist = get_acc_prob_hist(dst_R_hist)
    diff_B_acc_prob = abs(np.tile(src_B_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_B_acc_prob_hist.reshape(1, 256))
    diff_G_acc_prob = abs(np.tile(src_G_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_G_acc_prob_hist.reshape(1, 256))
    diff_R_acc_prob = abs(np.tile(src_R_acc_prob_hist.reshape(256, 1), (1, 256)) - dst_R_acc_prob_hist.reshape(1, 256))
    table_B = np.argmin(diff_B_acc_prob, axis=0).astype(np.int8)
    table_G = np.argmin(diff_G_acc_prob, axis=0).astype(np.int8)
    table_R = np.argmin(diff_R_acc_prob, axis=0).astype(np.int8)
    sp_B = cv2.LUT(dst_B, table_B)
    sp_G = cv2.LUT(dst_G, table_G)
    sp_R = cv2.LUT(dst_R, table_R)
    sp_image = cv2.merge([sp_B, sp_G, sp_R])
    return sp_image


img_dir = 'H:\\workspace\\luotongan\\SavedCamData\\test13\\img'
sp_img_dir = 'H:\\workspace\\luotongan\\SavedCamData\\test13\\sp2img'

if not os.path.exists(sp_img_dir):
    os.mkdir(sp_img_dir)

img_names = os.listdir(img_dir)
tmp_img = None

for img_name in tqdm.tqdm(img_names):
    img_file = os.path.join(img_dir, img_name)
    img = cv2.imread(img_file)
    if tmp_img is None:
        tmp_img = img
        sp2_img = img
    else:
        sp2_img = specify_hist_color_hsv(tmp_img, img)

    spe_img_file = os.path.join(sp_img_dir, img_name)
    cv2.imwrite(spe_img_file, sp2_img)