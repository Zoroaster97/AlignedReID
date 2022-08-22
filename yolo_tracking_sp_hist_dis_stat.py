import os, cv2, torch, tqdm, math
from pytorchyolo import detect, models
from aligned_reid.model.Model import Model
import torchvision.transforms as T
import numpy as np

IMG_H = 480
# IMG_W = 640
IMG_W = 890

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input      
    """
    x = 100. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
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


# Load the YOLO model
model = models.load_model(
  "yolocfg\\yolov3.cfg", 
  "yoloweights\\yolov3.weights")
#   "yolocfg\\yolov3-tiny.cfg", 
#   "yoloweights\\yolov3-tiny.weights")

reid_model = Model(local_conv_out_channels=128)
ckpt = torch.load('ckpts/model_weight.pth')
reid_model.load_state_dict(ckpt, strict=False)
reid_model.cuda().eval()

trf = T.Compose([
    # T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225])
])

# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658726098997\\img"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658979632557\\img"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658979342960\\img"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658979152063\\img"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658725708283\\cam1"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test1658725823165\\cam1"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test4\\img"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test4\\spimg"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test4\\spimg2"
# imgdir = "H:\\workspace\\luotongan\\Annotating\\test8\\img"
imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test16\\img"
img_files = os.listdir(imgdir)
init = False
idx = 0
init_feat = None
with torch.no_grad():
    while not init and idx < len(img_files):
        init_img = cv2.imread(os.path.join(imgdir, img_files[idx]))   # h, w, c
        init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        boxes = detect.detect_image(model, init_img)
        x1, y1, x2, y2, conf, cls = boxes[0]    # max conf while min cls (person)
        x1, y1, x2, y2 = max(1, x1), max(1, y1), min((IMG_W - 1), x2), min((IMG_H - 1), y2)
        idx += 1
        if cls == 0:
            # print(x2 - x1, y2 - y1)
            # tmp_img = torch.tensor(init_img[int(y1): int(y2), int(x1): int(x2), :], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            tmp_img = (init_img[int(y1): int(y2), int(x1): int(x2), :])
            tmp_img = cv2.resize(tmp_img,(128, 256))
            template_image = tmp_img
            tmp_img.swapaxes(0, 2)
            tmp_img.swapaxes(1, 2)
            tmp_img = trf(tmp_img).unsqueeze(0).cuda()
            # tmp_img = torch.tensor(tmp_img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            # print(tmp_img)
            # print(tmp_img.shape)    # torch.Size([1, 3, 256, 128])
            init_feat, _ = reid_model(tmp_img)
            init_feat = normalize(init_feat)
            # print(init_feat.shape)  # torch.Size([1, 512])
            init = True
            break

assert init
src_acc_prob_hist = get_hist_color_hsv(template_image)
template_image = cv2.resize(template_image,(64, 128))
template_image = cv2.cvtColor(template_image, cv2.COLOR_RGB2BGR)

# a = torch.rand(1,512).cuda()
# print(EuclideanDistances(a, init_feat))
# print(EuclideanDistances(init_feat, a))
min_dis_list = []
all_dis_list = []

with torch.no_grad():
    for i in tqdm.tqdm(range(idx, len(img_files))):
        p_imgs = []
        sc_img = cv2.imread(os.path.join(imgdir, img_files[i]))
        sc_img = cv2.cvtColor(sc_img, cv2.COLOR_BGR2RGB)
        # boxes = detect.detect_image(model, sc_img)
        boxes = detect.detect_image(model, sc_img, conf_thres=0.5)
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = max(1, x1), max(1, y1), min((IMG_W - 1), x2), min((IMG_H - 1), y2)
            if cls == 0:
                # p_img = torch.tensor(sc_img[int(y1): int(y2), int(x1): int(x2), :], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                p_img = (sc_img[int(y1): int(y2), int(x1): int(x2), :])
                # specifiy hist
                p_img = specify_hist_color_hsv(src_acc_prob_hist, p_img)
                p_img = cv2.resize(p_img,(128, 256))
                p_img.swapaxes(0, 2)
                p_img.swapaxes(1, 2)
                p_img = trf(p_img).unsqueeze(0).cuda()
                # p_img = torch.tensor(p_img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                p_imgs.append(p_img)
                new_boxes.append(box)
        boxes = new_boxes
        if len(p_imgs) > 1:
            min_dis = float('inf')
            all_dis = []
            for idx, p_img in enumerate(p_imgs):
                p_feat, _ = reid_model(p_img)
                p_feat = normalize(p_feat)
                dis = EuclideanDistances(init_feat, p_feat)
                dis = dis.item()
                all_dis.append(dis)
                # print(dis)
                if dis < min_dis:
                    min_dis = dis
            min_dis_list.append(min_dis)
            all_dis_list.append(all_dis)

print(len(min_dis_list))
assert len(min_dis_list) == len(all_dis_list)

dis_dis_mean = []
for i in range(len(min_dis_list)):
    min_dis = min_dis_list[i]
    all_dis = all_dis_list[i]
    cur_dis = 0
    for dis in all_dis:
        cur_dis += (dis - min_dis)
    cur_dis /= len(all_dis)
    dis_dis_mean.append(cur_dis)

min_dis_mean = sum(min_dis_list) / len(min_dis_list)
dis_dis_mean_mean = sum(dis_dis_mean) / len(dis_dis_mean)

assert len(min_dis_list) == len(dis_dis_mean)

min_dis_stdv = 0
dis_dis_mean_stdv = 0
dis_dis_stdv_list = []
for i in range(len(min_dis_list)):
    min_dis = min_dis_list[i]
    all_dis = all_dis_list[i]
    cur_dis_mean = dis_dis_mean[i]
    min_dis_stdv += (min_dis - min_dis_mean)**2
    dis_dis_mean_stdv += (cur_dis_mean - dis_dis_mean_mean)**2
    dis_dis_stdv = 0
    for dis in all_dis:
        cur_dis = (dis - min_dis)
        dis_dis_stdv += (cur_dis - cur_dis_mean)**2
    if len(all_dis) > 0:
        dis_dis_stdv /= len(all_dis)
        dis_dis_stdv = math.sqrt(dis_dis_stdv)
        dis_dis_stdv_list.append(dis_dis_stdv)

min_dis_stdv /= len(min_dis_list)
min_dis_stdv = math.sqrt(min_dis_stdv)
dis_dis_mean_stdv /= len(dis_dis_mean)
dis_dis_mean_stdv = math.sqrt(dis_dis_mean_stdv)

dis_dis_stdv_mean = sum(dis_dis_stdv_list) / len(dis_dis_stdv_list)
dis_dis_stdv_stdv = 0
for dis_dis_stdv in dis_dis_stdv_list:
    dis_dis_stdv_stdv += (dis_dis_stdv - dis_dis_stdv_mean)**2
dis_dis_stdv_stdv /= len(dis_dis_stdv_list)
dis_dis_stdv_stdv = math.sqrt(dis_dis_stdv_stdv)

print('mdm:', min_dis_mean)
print('mds', min_dis_stdv)

print('ddmm:', dis_dis_mean_mean)
print('ddms:', dis_dis_mean_stdv)

print('ddsm:', dis_dis_stdv_mean)
print('ddss:', dis_dis_stdv_stdv)
