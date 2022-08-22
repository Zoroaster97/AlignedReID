import cv2, os, tqdm
import numpy as np


def equalize_hist_color_hsv(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_v = cv2.equalizeHist(v)
    eq_image = cv2.cvtColor(cv2.merge([h, s, eq_v]), cv2.COLOR_HSV2BGR)
    return eq_image

def equalize_hist_color_yuv(img):
    y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eq_y = cv2.equalizeHist(y)
    eq_image = cv2.cvtColor(cv2.merge([eq_y, u, v]), cv2.COLOR_YUV2BGR)
    return eq_image

def equalize_hist_color_lab(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    eq_l = cv2.equalizeHist(l)
    eq_image = cv2.cvtColor(cv2.merge([eq_l, a, b]), cv2.COLOR_LAB2BGR)
    return eq_image

def equalize_hist_color_bgr(img):
    b, g, r = cv2.split(img)
    eq_b = cv2.equalizeHist(b)
    eq_g = cv2.equalizeHist(g)
    eq_r = cv2.equalizeHist(r)
    eq_image = cv2.merge([eq_b, eq_g, eq_r])
    return eq_image


cam1_img_dir = 'H:\\workspace\\luotongan\\SavedCamData\\test13\\cam1'
cam2_img_dir = 'H:\\workspace\\luotongan\\SavedCamData\\test13\\cam2'
eq_img_dir = 'H:\\workspace\\luotongan\\SavedCamData\\test13\\eqimg'

if not os.path.exists(eq_img_dir):
    os.mkdir(eq_img_dir)

img_names = os.listdir(cam1_img_dir)

for img_name in tqdm.tqdm(img_names):
    cam1_img_file = os.path.join(cam1_img_dir, img_name)
    cam2_img_file = os.path.join(cam2_img_dir, img_name)

    cam1_img = cv2.imread(cam1_img_file)
    cam2_img = cv2.imread(cam2_img_file)

    # eq_cam1_img = equalize_hist_color_hsv(cam1_img)
    # eq_cam2_img = equalize_hist_color_hsv(cam2_img)
    eq_cam1_img = equalize_hist_color_bgr(cam1_img)
    eq_cam2_img = equalize_hist_color_bgr(cam2_img)

    cat_eq_img = np.zeros((480, 890, 3), dtype=np.uint8)
    cat_eq_img[:, 0:445, :] = eq_cam1_img[:, 0:445, :]
    cat_eq_img[:, 445:890, :] = eq_cam2_img[:, 195:640, :]
    eq_img_file = os.path.join(eq_img_dir, img_name)
    cv2.imwrite(eq_img_file, cat_eq_img)