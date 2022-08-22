import os, cv2, torch
from pytorchyolo import detect, models
from aligned_reid.model.Model import Model
import torchvision.transforms as T
import numpy as np
import time, tqdm
from yolo_eval_utils import Evaluater, calc_curves, calc_metrics, normalize, get_hist_color_hsv, specify_hist_color_hsv, EuclideanDistances, NBINS_IOU


IMG_H = 480
IMG_W = 890

# Load the YOLO model
model = models.load_model(
  "yolocfg\\yolov3.cfg", 
  "yoloweights\\yolov3.weights")
#   "yolocfg\\yolov3-tiny.cfg", 
#   "yoloweights\\yolov3-tiny.weights")

reid_model = Model(local_conv_out_channels=128)
ckpt = torch.load('ckpts/model_weight.pth')
# ckpt = torch.load('ckpts/model_weight_ml.pth')
reid_model.load_state_dict(ckpt, strict=False)
reid_model.cuda().eval()

trf = T.Compose([
    # T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225])
])

f = open('eval.txt', 'w')

# imgdir = "H:\\workspace\\luotongan\\SavedCamData\\test13\\img"
for sample_id in range(1, 21):
    imgdir = "H:\\workspace\\luotongan\\CamData-Datasets\\CamData1\\person\\person-{}\\img".format(sample_id)
    print('person-{}'.format(sample_id))
    f.write('person-{}\n'.format(sample_id))

    # evaluater = Evaluater(os.path.dirname(imgdir))
    evaluater = Evaluater(os.path.dirname(imgdir), 'groundtruth.txt', 0.5)
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
    print('start idx:', i)
    b = []
    a = []
    tic = time.time()
    with torch.no_grad():
        # while i < len(img_files):
        i0 = i
        for i in tqdm.tqdm(range(i0, len(img_files))):
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
            # if len(p_imgs) > 1:
            #     evaluater.eval(i, [x1, y1, x2 - x1, y2 - y1])
            # evaluater.eval(i, [x1, y1, x2 - x1, y2 - y1])
            b.append([x1, y1, x2 - x1, y2 - y1])
            a.append(evaluater.annos[i])
            # i += 1
    toc = time.time()
    print('end idx:', i - 1)
    print('cost time:', toc - tic)
    print('FPS:', i / (toc - tic))

    # result = evaluater.get_eval_result()
    # for k in result:
    #     print(k, result[k])
    b = np.array(b)
    a = np.array(a)
    assert len(a) == len(b)

    ious, center_errors, norm_center_errors = calc_metrics(b, a)
    succ_curve, prec_curve, norm_prec_curve = calc_curves(ious, center_errors, norm_center_errors)

    print('success_score:', np.mean(succ_curve))
    print('precision_score:', prec_curve[20])
    print('normalized_precision_score:', np.mean(norm_prec_curve),)
    print('success_rate:', succ_curve[NBINS_IOU // 2])

    f.write('success_score:: {}\n'.format(np.mean(succ_curve)))
    f.write('precision_score:: {}\n'.format(prec_curve[20]))
    f.write('normalized_precision_score:: {}\n'.format(np.mean(norm_prec_curve),))
    f.write('success_rate: {}\n'.format(succ_curve[NBINS_IOU // 2]))
    f.write('\n')

f.close()

