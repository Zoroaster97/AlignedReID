

class Config:
    def __init__(self) -> None:
        # show, eval, stat, save_video
        self.MODE = 'save_video'
        self.IMG_H = 480
        self.IMG_W = 890
        # self.IMG_W = 640

        # self.SAVE_VIDEO = True
        self.SAVE_VIDEO_NAME = 'test16.mp4'
        self.SAVE_VIDEO_FPS = 10


        # yolo
        self.YOLO_CFG = "yolocfg\\yolov3.cfg"
        self.YOLO_WEIGHTS = "yoloweights\\yolov3.weights"
        # reid
        self.REID_WEIGHTS = 'ckpts/model_weight.pth'
        # specify color hist
        self.SP_HIST = False
        # refresh template
        self.REFRESH = True
        self.REFRESH_STEP = 1
        # self.REFRESH_THRESHOLD = 30
        self.REFRESH_THRESHOLD = 33
        # self.REFRESH_THRESHOLD = 40
        self.REFRESH_SAVETY_DIS = 1
        # self.REFRESH_INTERW_THRES = 0.3
        self.REFRESH_INTERW_THRES = 0.2
        # none, semi
        self.REFRESH_DYNAMIC = 'semi'
        # filtering small object
        self.FILTERING_SMALL_OBJ = False
        self.SMALL_OBJ_THRES = 6000
        # filtering short object
        self.FILTERING_SHORT_OBJ = True
        self.SHORT_OBJ_THRES = 120
        # filtering distant object  NOTICE: turn on cautiously with refresh template
        self.FILTERING_DISTANT_OBJ = False
        self.DISTANT_OBJ_THRES = 200




        # momentum updating tracking box
        self.TRACKING_BOX_MOME = False
        self.TRUST_DIS_THRES = 20
        self.DISTRUST_DIS_DIFF_THRES = 24
        # linear, semi, sigmoid
        self.MOME_MODE = 'sigmoid'
        self.MOME_LOG_SCALE = 2.

        # show
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test19\\img"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test14\\img"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test20\\img"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test20\\img2"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test8\\img"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test8"
        self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test16"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test20"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test24"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\SavedCamData\\test8\\cam1"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\CamData-Datasets\\CamData1\\person\\person-15\\img"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\CamData-Datasets\\CamData2\\person\\person-8\\img"
        # self.IMG_DIR = "H:\\workspace\\luotongan\\CamData-Datasets\\CamData3\\person\\person-17\\img"

        # eval
        # self.EVAL_IMGDIR = "H:\\workspace\\luotongan\\CamData-Datasets\\CamData1\\person\\person-*\\img"
        # self.EVAL_SAMPLE_IDS = range(1, 21)
        # self.EVAL_IMGDIR = "H:\\workspace\\luotongan\\CamData-Datasets\\CamData2\\person\\person-*\\img"
        # self.EVAL_SAMPLE_IDS = [1, 2, 3, 4, 5, 6, 8]
        self.EVAL_IMGDIR = "H:\\workspace\\luotongan\\CamData-Datasets\\CamData3\\person\\person-*\\img"
        self.EVAL_SAMPLE_IDS = range(12, 19)
        self.EVAL_SAVE_NAME = 'eval_test.txt'
        # self.EVAL_SAVE_NAME = 'eval2_sphist.txt'
        # self.EVAL_SAVE_NAME = 'eval3_sphist.txt'
        # self.EVAL_SAVE_NAME = 'eval_sphist_refresh16.txt'
        # self.EVAL_SAVE_NAME = 'eval2_sphist_refresh16.txt'
        # self.EVAL_SAVE_NAME = 'eval2_sphist_refresh20.txt'
        # self.EVAL_SAVE_NAME = 'eval_sphist_refresh30.txt'
        # self.EVAL_SAVE_NAME = 'eval2_sphist_refresh30.txt'
        # self.EVAL_SAVE_NAME = 'eval2_sphist_refresh33_2.txt'
        # self.EVAL_SAVE_NAME = 'eval3_sphist_refresh30.txt'
        # self.EVAL_SAVE_NAME = 'eval3_sphist_refresh30_filtering6000.txt'
        # self.EVAL_SAVE_NAME = 'eval_sphist_filtering5000.txt'
        # self.EVAL_SAVE_NAME = 'eval_sphist_filtering6000.txt'
        # self.EVAL_SAVE_NAME = 'eval2_sphist_filtering6000.txt'
        # self.EVAL_SAVE_NAME = 'eval3_sphist_filtering6000.txt'
        # self.EVAL_SAVE_NAME = 'eval_sphist_filtering7000.txt'
        # self.EVAL_SAVE_NAME = 'eval2_refresh33_2.txt'
        # self.EVAL_SAVE_NAME = 'eval3_refresh33_2_filtering6000.txt'
        # self.GT_NAME = 'groundtruth.txt'
        self.GT_NAME = 'yolo_anno.txt'
        self.SUCC_IOU = 0.5

