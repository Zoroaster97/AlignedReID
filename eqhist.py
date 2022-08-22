import cv2
import numpy as np

cam1_img_file = 'H:\\workspace\\luotongan\\SavedCamData\\test2\\cam1\\1658308105574.jpg'
cam2_img_file = 'H:\\workspace\\luotongan\\SavedCamData\\test2\\cam2\\1658308105574.jpg'
cat_img_file = 'H:\\workspace\\luotongan\\SavedCamData\\test2\\img\\1658308105574.jpg'

cam1_img = cv2.imread(cam1_img_file)
cam2_img = cv2.imread(cam2_img_file)
cat_img = cv2.imread(cat_img_file)

def equalize_hist_color_hsv(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image

def equalize_hist_color_hsv_full(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR_FULL)
    return eq_image

def equalize_hist_color_yuv(img):
    Y, U, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eq_Y = cv2.equalizeHist(Y)
    eq_image = cv2.cvtColor(cv2.merge([eq_Y, U, V]), cv2.COLOR_YUV2BGR)
    return eq_image

def equalize_hist_color_lab(img):
    L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    eq_L = cv2.equalizeHist(L)
    eq_image = cv2.cvtColor(cv2.merge([eq_L, A, B]), cv2.COLOR_LAB2BGR)
    return eq_image

eq_cat_img = equalize_hist_color_hsv(cat_img)
cv2.imwrite('cat.jpg', cat_img)
cv2.imwrite('eq_cat.jpg', eq_cat_img)

eq_cam1_img = equalize_hist_color_hsv(cam1_img)
eq_cam2_img = equalize_hist_color_hsv(cam2_img)
cat_eq_img = np.zeros((480, 890, 3), dtype=np.uint8)
cat_eq_img[:, 0:445, :] = eq_cam1_img[:, 0:445, :]
cat_eq_img[:, 445:890, :] = eq_cam2_img[:, 195:640, :]
cv2.imwrite('cat_eq.jpg', cat_eq_img)