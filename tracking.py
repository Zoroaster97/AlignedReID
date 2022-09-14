import os, cv2, torch, copy, time, tqdm, math
from pytorchyolo import detect, models
from aligned_reid.model.Model import Model
import torchvision.transforms as T
import numpy as np
from utils import normalize, get_hist_color_hsv, specify_hist_color_hsv, EuclideanDistances
from yolo_eval_utils import Evaluater, calc_curves, calc_metrics, NBINS_IOU
from cfg import Config


class Tracker:
    def __init__(self, cfg: Config) -> None:
        # Load the YOLO model
        self.yolo_model = models.load_model(cfg.YOLO_CFG, cfg.YOLO_WEIGHTS)
        self.reid_model = Model(local_conv_out_channels=128)
        ckpt = torch.load(cfg.REID_WEIGHTS)
        self.reid_model.load_state_dict(ckpt, strict=False)
        self.reid_model.cuda().eval()
        self.trf = T.Compose([
            # T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225])
        ])
        self.cfg = cfg
        self.template_idx = 0
        self.template_feat = None
        self.template_acc_prob_hist = None
        self.template_image = None

        self.img_files = None

        self.tracking_box = None
        self.init_refresh_thres = cfg.REFRESH_THRESHOLD
    

    def process_img(self, img):
        img = cv2.resize(img,(128, 256))
        img.swapaxes(0, 2)
        img.swapaxes(1, 2)
        img = self.trf(img).unsqueeze(0).cuda()
        return img
    

    def get_feats(self, p_imgs: list):
        imgs = copy.deepcopy(p_imgs)
        for i, img in enumerate(imgs):
            imgs[i] = self.process_img(img)
        with torch.no_grad():
            feats, _ = self.reid_model(torch.cat(imgs))
        return normalize(feats, scale=100)
    

    def process_box(self, box):
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = max(1, x1), max(1, y1), min((self.cfg.IMG_W - 1), x2), min((self.cfg.IMG_H - 1), y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return x1, y1, x2, y2, conf, cls


    def init_feat(self, img_dir=None):
        if self.cfg.MODE == 'show':
            self.img_files = os.listdir(self.cfg.IMG_DIR)
            img_dir = self.cfg.IMG_DIR
        elif self.cfg.MODE == 'eval':
            assert self.img_files is not None
            assert img_dir is not None
        # i: init_idx
        i = 0
        while i < len(self.img_files):
            init_img = cv2.imread(os.path.join(img_dir, self.img_files[i]))    # h, w, c
            init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
            boxes = detect.detect_image(self.yolo_model, init_img)
            box = boxes[0]  # max conf while min cls (person)
            x1, y1, x2, y2, conf, cls = self.process_box(box)
            i += 1
            if cls == 0:
                template_img = init_img[y1: y2, x1: x2, :]
                self.template_feat = self.get_feats([template_img])
                if self.cfg.SP_HIST:
                    self.template_acc_prob_hist = get_hist_color_hsv(template_img)
                template_img = cv2.resize(template_img,(64, 128))
                template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2BGR)
                self.template_image = template_img
                self.template_idx = i - 1
                self.tracking_box = [x1, y1, x2, y2]
                break
        if i == len(self.img_files):
            i = -1
        return i
    

    def refresh_template(self, context, step=1, dis_threshold=20):
        if (context['img_idx'] - self.template_idx >= step  or context['img_idx'] - self.template_idx <= 0)\
            and context['dis'] < dis_threshold:
            self.template_feat = context['feat'].unsqueeze(0)
            self.template_idx = context['img_idx']
            template_image = context['p_img']
            if self.cfg.SP_HIST:
                self.template_acc_prob_hist = get_hist_color_hsv(template_image)
            template_image = cv2.resize(template_image, (64, 128))
            self.template_image = cv2.cvtColor(template_image, cv2.COLOR_RGB2BGR)
            if self.cfg.REFRESH_DYNAMIC == 'semi':
                if context['sec_dis'] > 0:
                    self.cfg.REFRESH_THRESHOLD = (context['dis'] + context['sec_dis']) / 2
                else:
                    self.cfg.REFRESH_THRESHOLD = (self.cfg.REFRESH_THRESHOLD + self.init_refresh_thres) / 2
    
    
    def checking_iou(self, tracking_box, boxes, thres=0.3):
        tx1, ty1, tx2, ty2 = tracking_box
        ts = (tx2 - tx1) * (ty2 - ty1)
        for box in boxes:
            if tracking_box == box:
                # print('yes!!!')
                continue
            x1, y1, x2, y2 = box
            ix1, iy1 = max(tx1, x1), max(ty1, y1)
            ix2, iy2 = min(tx2, x2), min(ty2, y2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inters = iw * ih
            s = (x2 - x1) * (y2 - y1)
            iou = inters / (ts + s - inters)
            if iou >= thres:
                return False
        return True
    
    def checking_interw(self, tracking_box, boxes, thres=0.3):
        tx1, ty1, tx2, ty2 = tracking_box
        tw = tx2 - tx1
        for box in boxes:
            if tracking_box == box:
                # print('yes!!!')
                continue
            x1, y1, x2, y2 = box
            ix1, ix2 = max(tx1, x1), min(tx2, x2)
            iw = max(0, ix2 - ix1)
            interw = iw / tw
            # print('----------------info----------------')
            # print(tx1, tx2, tw)
            # print(x1, x2)
            # print(ix1, ix2, iw, interw)
            if interw >= thres:
                return False
        return True
    

    def tracking_inference(self, img, i):
        p_imgs = []
        raw_p_imgs = []
        p_boxes = []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = detect.detect_image(self.yolo_model, img, conf_thres=0.5)
        for box in boxes:
            x1, y1, x2, y2, conf, cls = self.process_box(box)
            if cls == 0:
                if self.cfg.FILTERING_SMALL_OBJ and (x2 - x1) * (y2 - y1) < self.cfg.SMALL_OBJ_THRES:
                    continue
                if self.cfg.FILTERING_DISTANT_OBJ and abs(self.tracking_box[0] + self.tracking_box[2] - 
                    x1 - x2) / 2 > self.cfg.DISTANT_OBJ_THRES:
                    continue
                p_img = img[y1: y2, x1: x2, :]
                raw_p_imgs.append(copy.deepcopy(p_img))
                if self.cfg.SP_HIST:
                    p_img = specify_hist_color_hsv(self.template_acc_prob_hist, p_img)
                p_imgs.append(p_img)
                p_boxes.append([x1, y1, x2, y2])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        dists = None
        min_dis_idx = -1
        if len(p_imgs) > 0:
            feats = self.get_feats(p_imgs)
            dists = EuclideanDistances(self.template_feat, feats)
            dists = dists.squeeze(0)
            min_dis_idx = torch.argmin(dists).item()
            sec_min_dis_idx = None
            if len(dists) > 1:
                sec_min_dis_idx = dists.kthvalue(2)[1].item()
            if self.cfg.TRACKING_BOX_MOME:
                delta_dis = dists[min_dis_idx].item() - self.cfg.TRUST_DIS_THRES
                # delta_dis = max(0, delta_dis)
                delta_dis = min(max(0, delta_dis), self.cfg.DISTRUST_DIS_DIFF_THRES)
                if self.cfg.MOME_MODE in ['linear', 'semi']:
                    # keep_rate: [0, 0.5]
                    if self.cfg.MOME_MODE == 'semi':
                        keep_rate = 0.5
                    else:
                        keep_rate = delta_dis / self.cfg.DISTRUST_DIS_DIFF_THRES / 2
                elif self.cfg.MOME_MODE == 'sigmoid':
                    delta_dis -= self.cfg.DISTRUST_DIS_DIFF_THRES
                    delta_dis *= self.cfg.MOME_LOG_SCALE
                    keep_rate = 1 / (1 + math.exp(-delta_dis / self.cfg.DISTRUST_DIS_DIFF_THRES))
                else:
                    raise NotImplementedError('Wrong MOME_MODE value! ')
                print(dists[min_dis_idx].item())
                print(keep_rate)
                cur_box = p_boxes[min_dis_idx]
                for bi in range(len(self.tracking_box)):
                    self.tracking_box[bi] = int(self.tracking_box[bi] * keep_rate + cur_box[bi] * (1 - keep_rate))
            else:
                self.tracking_box = p_boxes[min_dis_idx]
            # if self.cfg.REFRESH:
            if self.cfg.REFRESH and self.tracking_box[0] > 5 and self.tracking_box[2] < (self.cfg.IMG_W - 5):
                if len(dists) > 1:
                    sec_dis = dists[sec_min_dis_idx].item()
                else:
                    sec_dis = -1.
                # if self.checking_iou(self.tracking_box, p_boxes, self.cfg.REFRESH_IOU_THRES):
                if self.checking_interw(self.tracking_box, p_boxes, self.cfg.REFRESH_INTERW_THRES):
                    context = {'dis': dists[min_dis_idx].item(), 'feat': feats[min_dis_idx], 
                                'img_idx': i, 'p_img': raw_p_imgs[min_dis_idx], 'sec_dis': sec_dis}
                    self.refresh_template(context, self.cfg.REFRESH_STEP, self.cfg.REFRESH_THRESHOLD)
        return img, p_imgs, p_boxes, dists, min_dis_idx
    

    def show(self):
        i = self.init_feat()
        if i == -1:
            print('No person has been detected. ')
            return
        while True:
            p_imgs = []
            p_boxes = []
            sc_img = cv2.imread(os.path.join(self.cfg.IMG_DIR, self.img_files[i]))
            sc_img, p_imgs, p_boxes, dists, min_dis_idx = self.tracking_inference(sc_img, i)
            sc_img[:128, :64, :] = self.template_image[:,:,:]
            if len(p_imgs) > 0:
                tracking_img = p_imgs[min_dis_idx]
                tracking_img = cv2.resize(tracking_img, (64, 128))
                tracking_img = cv2.cvtColor(tracking_img, cv2.COLOR_RGB2BGR)
                sc_img[:128, 64:128, :] = tracking_img
                for pidx, box in enumerate(p_boxes):
                    x1, y1, x2, y2 = box
                    color = (255, 255, 100) if pidx == min_dis_idx else (0, 255, 0)
                    dis = dists[pidx].item()
                    cv2.rectangle(sc_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(sc_img, '%.3f' % (dis), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
                tx1, ty1, tx2, ty2 = self.tracking_box
                cv2.rectangle(sc_img, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
            if self.cfg.REFRESH:
                cv2.putText(sc_img, '%.3f' % (self.cfg.REFRESH_THRESHOLD), (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.imshow('img', sc_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('w') and i < len(self.img_files) - 1:
                i += 1
            elif key == ord('e'):
                print('w:', tx2 - tx1)
                print('h:', ty2 - ty1)
                print('s:', (tx2 - tx1) * (ty2 - ty1))
            elif key == ord('d'):
                print('x1:', tx1)
                print('y1:', ty1)
                print('x2:', tx2)
                print('y2:', ty2)
            elif key == ord('s') and i > 0:
                i -= 1
            elif key == ord('a'):
                print(i)
            # elif key == ord('d'):
            #     try:
            #         new_idx = int(input('enter new idx: '))
            #         if 0 < new_idx < len(self.img_files):
            #             i = new_idx
            #     except Exception as e:
            #         print(e)
            elif key == ord('f'):
                cv2.imwrite('sample_{}.jpg'.format(i), sc_img)
    

    def eval(self):
        f = open(self.cfg.EVAL_SAVE_NAME, 'w')
        for sample_id in self.cfg.EVAL_SAMPLE_IDS:
            imgdir = self.cfg.EVAL_IMGDIR.replace('*', str(sample_id))
            print('person-{}'.format(sample_id))
            f.write('person-{}\n'.format(sample_id))
            evaluater = Evaluater(os.path.dirname(imgdir), self.cfg.GT_NAME, self.cfg.SUCC_IOU)
            self.img_files = os.listdir(imgdir)
            init_idx = self.init_feat(imgdir)
            if init_idx == -1:
                print('No person has been detected. ')
                return
            print('start idx:', init_idx)
            b = []
            a = []
            tic = time.time()
            for i in tqdm.tqdm(range(init_idx, len(self.img_files))):
                sc_img = cv2.imread(os.path.join(imgdir, self.img_files[i]))
                sc_img, p_imgs, p_boxes, dists, min_dis_idx = self.tracking_inference(sc_img, i)
                if len(p_boxes) > 0:
                    x1, y1, x2, y2 = p_boxes[min_dis_idx]
                else: 
                    x1, y1, x2, y2 = 0, 0, 0, 0
                b.append([x1, y1, x2 - x1, y2 - y1])
                a.append(evaluater.annos[i])
            toc = time.time()
            print('end idx:', i - 1)
            print('cost time:', toc - tic)
            print('FPS:', i / (toc - tic))

            b = np.array(b)
            a = np.array(a)
            assert len(a) == len(b)

            ious, center_errors, norm_center_errors = calc_metrics(b, a)
            succ_curve, prec_curve, norm_prec_curve = calc_curves(ious, center_errors, norm_center_errors)

            print('success_score:', np.mean(succ_curve))
            print('precision_score:', prec_curve[20])
            print('normalized_precision_score:', np.mean(norm_prec_curve),)
            print('success_rate:', succ_curve[NBINS_IOU // 2])

            f.write('success_score: {}\n'.format(np.mean(succ_curve)))
            f.write('precision_score: {}\n'.format(prec_curve[20]))
            f.write('normalized_precision_score: {}\n'.format(np.mean(norm_prec_curve),))
            f.write('success_rate: {}\n'.format(succ_curve[NBINS_IOU // 2]))
            f.write('\n')
            f.flush()

        f.close()


    def run(self):
        if self.cfg.MODE == 'show':
            self.show()
        elif self.cfg.MODE == 'eval':
            self.eval()
        else:
            pass


if __name__ == '__main__':
    cfg = Config()
    tracker = Tracker(cfg)
    tracker.run()
