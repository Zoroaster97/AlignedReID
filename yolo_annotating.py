import os, cv2, torch
from pytorchyolo import detect, models
from aligned_reid.model.Model import Model
import torchvision.transforms as T
import numpy as np

class Annotater:
    def __init__(self, dirname, filename='yolo_anno.txt'):
        self.anno_file_name = os.path.join(dirname, filename)
        self.annos = []
        self.str_annos = []
        if os.path.exists(self.anno_file_name):
            self.load_annos()
    
    def load_annos(self):
        with open(self.anno_file_name) as f:
            self.str_annos = f.readlines()
        self.annos = []
        for str_anno in self.str_annos:
            x, y, w, h = str_anno.strip().split(',')
            self.annos.append([int(x), int(y), int(w), int(h)])

    def dump_annos(self):
        with open(self.anno_file_name, 'w') as f:
            f.writelines(self.str_annos)
    
    def annotate(self, idx, box):
        assert idx <= len(self.annos) == len(self.str_annos)
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        str_box = '{},{},{},{}\n'.format(x, y, w, h)
        if idx == len(self.annos):
            self.annos.append([x, y, w, h])
            self.str_annos.append(str_box)
        else:
            self.annos[idx] = [x, y, w, h]
            self.str_annos[idx] = str_box


IMG_H = 480
IMG_W = 890

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
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test7\\img"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test7\\skimg"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test4\\img"
# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test4\\spimg"
# imgdir = "H:\\workspace\\luotongan\\Annotating\\test8\\img"
imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test16\\img"
annotater = Annotater(os.path.dirname(imgdir))
img_files = os.listdir(imgdir)
init = False
i = 0
init_feat = None
with torch.no_grad():
    while not init and i < len(img_files):
        init_img = cv2.imread(os.path.join(imgdir, img_files[i]))   # h, w, c
        init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        boxes = detect.detect_image(model, init_img)
        x1, y1, x2, y2, conf, cls = boxes[0]    # max conf while min cls (person)
        # x1, y1, x2, y2, conf, cls = boxes[2]    # for test18
        x1, y1, x2, y2 = max(1, x1), max(1, y1), min((IMG_W - 1), x2), min((IMG_H - 1), y2)
        annotater.annotate(i, [x1, y1, x2 - x1, y2 - y1])
        i += 1
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

# a = torch.rand(1,512).cuda()
# print(EuclideanDistances(a, init_feat))
# print(EuclideanDistances(init_feat, a))

with torch.no_grad():
    while True:
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
                p_img = specify_hist_color_hsv(src_acc_prob_hist, p_img)
                p_img = cv2.resize(p_img,(128, 256))
                p_img.swapaxes(0, 2)
                p_img.swapaxes(1, 2)
                p_img = trf(p_img).unsqueeze(0).cuda()
                # p_img = torch.tensor(p_img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                p_imgs.append(p_img)
                new_boxes.append(box)
        boxes = new_boxes
        if len(p_imgs) == 0:
            tracking_box = [0, 0, 0, 0, 0, 0]
        else: 
            tracking_box = boxes[0]
        if len(p_imgs) > 1:
            # p_imgs = torch.cat(p_imgs)  # shape(h, w) need to be aligned
            min_dis = float('inf')
            md_idx = 0
            # print('CMP')
            for idx, p_img in enumerate(p_imgs):
                p_feat, _ = reid_model(p_img)
                p_feat = normalize(p_feat)
                dis = EuclideanDistances(init_feat, p_feat)
                dis = dis.item()
                cur_box = boxes[idx]
                cx1, cy1, cx2, cy2, _, _ = cur_box
                cx1, cy1, cx2, cy2 = max(1, cx1), max(1, cy1), min((IMG_W - 1), cx2), min((IMG_H - 1), cy2)
                cv2.rectangle(sc_img, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0,255,0), 2)
                cv2.putText(sc_img, '%.3f' % (dis*100), (int(cx1), int(cy1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(sc_img, str(idx), (int(cx1), int(cy1)+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                # print(dis)
                if dis < min_dis:
                    min_dis = dis
                    md_idx = idx
                    # print(idx)
            tracking_box = boxes[md_idx]
        x1, y1, x2, y2, conf, cls = tracking_box
        x1, y1, x2, y2 = max(1, x1), max(1, y1), min((IMG_W - 1), x2), min((IMG_H - 1), y2)

        sc_img = cv2.cvtColor(sc_img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(sc_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
        if i < len(annotater.annos):
            xa, ya, wa, ha = annotater.annos[i]
            cv2.rectangle(sc_img, (xa, ya), (xa + wa, ya + ha), (0,0,255), 2)
        else:
            annotater.annotate(i, [x1, y1, x2 - x1, y2 - y1])
        cv2.imshow('img', sc_img)
        key = cv2.waitKey()
        if key == ord('q'):
            annotater.dump_annos()
            print('Saved')
            break
        elif key == ord('w') and i < len(img_files) - 1:
            i += 1
        elif key == ord('s') and i > 0:
            i -= 1
        elif key == ord('a'):
            print(i)
        elif key == ord('d'):
            annotater.dump_annos()
            print('Saved')
        # elif key == ord('f'):
        #     cv2.imwrite('sample_{}.jpg'.format(i), sc_img)
        elif key <= ord('9') and key >= ord('0'):
            # print(key)
            cidx = key - ord('0')
            if cidx < len(boxes):
                cx1, cy1, cx2, cy2, _, _ = boxes[cidx]
                cx1, cy1, cx2, cy2 = max(1, cx1), max(1, cy1), min((IMG_W - 1), cx2), min((IMG_H - 1), cy2)
                annotater.annotate(i, [cx1, cy1, cx2 - cx1, cy2 - cy1])
            else:
                print('Idx out of range')