import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List
from skimage.io import imsave
from network.pretrain_models import VGGBNPretrain
from utils.base_utils import color_map_forward, transformation_crop, to_cpu_numpy,color_map_backward
from utils.bbox_utils import parse_bbox_from_scale_offset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
class BaseDetector(nn.Module):
    def load_impl(self, ref_imgs):
        raise NotImplementedError

    def detect_impl(self, que_imgs):
        raise NotImplementedError

    def load(self, ref_imgs):
        ref_imgs = torch.from_numpy(color_map_forward(ref_imgs)).permute(0, 3, 1, 2).cuda()
        self.load_impl(ref_imgs)

    def detect(self, que_imgs):
        que_imgs = torch.from_numpy(color_map_forward(que_imgs)).permute(0, 3, 1, 2).cuda()
        return self.detect_impl(que_imgs) # 'scores' 'select_pr_offset' 'select_pr_scale'

    @staticmethod
    def parse_detect_results(results):
        """

        @param results: dict
            pool_ratio: int -- pn
            scores: qn,1,h/pn,w/pn
            select_pr_offset: qn,2,h/pn,w/pn
            select_pr_scale:  qn,1,h/pn,w/pn
            select_pr_angle:  qn,2,h/pn,w/pn # optional
        @return: all numpy ndarray
        """
        qn = results['scores'].shape[0]
        pool_ratio = results['pool_ratio']

        # max scores
        _, score_x, score_y = BaseDetector.get_select_index(results['scores']) # qn
        position = torch.stack([score_x, score_y], -1)  # qn,2

        # offset
        offset = results['select_pr_offset'][torch.arange(qn),:,score_y,score_x] # qn,2
        position = position + offset

        # to original coordinate
        position = (position + 0.5) * pool_ratio - 0.5 # qn,2

        # scale
        scale_r2q = results['select_pr_scale'][torch.arange(qn),0,score_y,score_x] # qn
        scale_r2q = 2**scale_r2q
        outputs = {'position': position.detach().cpu().numpy(), 'scale_r2q': scale_r2q.detach().cpu().numpy()}
        # rotation
        if 'select_pr_angle' in results:
            angle_r2q = results['select_pr_angle'][torch.arange(qn),:,score_y,score_x] # qn,2
            angle = torch.atan2(angle_r2q[:,1],angle_r2q[:,0])
            outputs['angle_r2q'] = angle.cpu().numpy() # qn
        return outputs

    @staticmethod
    def detect_results_to_bbox(dets, length):
        pos = dets['position'] # qn,2
        length = dets['scale_r2q'] * length # qn,
        length = length[:,None]
        begin = pos - length/2
        return np.concatenate([begin,length,length],1)

    @staticmethod
    def detect_results_to_image_region(imgs, dets, region_len):
        qn = len(imgs)
        img_regions = []
        for qi in range(qn):
            pos = dets['position'][qi]; scl_r2q = dets['scale_r2q'][qi]
            ang_r2q = dets['angle_r2q'][qi] if 'anlge_r2q' in dets else 0
            img = imgs[qi]
            img_region, _ = transformation_crop(img, pos, 1/scl_r2q, -ang_r2q, region_len)
            img_regions.append(img_region)
        return img_regions

    @staticmethod
    def get_select_index(scores):
        """
        @param scores: qn,rfn or 1,hq,wq
        @return: qn
        """
        qn, rfn, hq, wq = scores.shape
        select_id = torch.argmax(scores.flatten(1), 1) # qn
        select_ref_id = select_id // (hq * wq)
        select_h_id = (select_id - select_ref_id * hq * wq) // wq
        select_w_id = select_id - select_ref_id * hq * wq - select_h_id * wq
        return select_ref_id, select_w_id, select_h_id

    @staticmethod
    def parse_detection(scores, scales, offsets, pool_ratio):
        """

        @param scores:    qn,1,h/8,w/8
        @param scales:    qn,1,h/8,w/8
        @param offsets:   qn,2,h/8,w/8
        @param pool_ratio:int
        @return: position in x_cur
        """
        qn, _, _, _ = offsets.shape

        _, score_x, score_y = BaseDetector.get_select_index(scores) # qn
        positions = torch.stack([score_x, score_y], -1)  # qn,2

        offset = offsets[torch.arange(qn),:,score_y,score_x] # qn,2
        positions = positions + offset

        # to original coordinate
        positions = (positions + 0.5) * pool_ratio - 0.5 # qn,2

        # scale
        scales = scales[torch.arange(qn),0,score_y,score_x] # qn
        scales = 2**scales
        return positions, scales # [qn,2] [qn]

def disable_bn_grad(input_module):
    for module in input_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)

def disable_bn_track(input_module):
    for module in input_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

class Detector(BaseDetector):
    default_cfg={
        "vgg_score_stats": [[36.264317,13.151907],[13910.291,5345.965],[829.70807,387.98788]],
        "vgg_score_max": 10,

        "detection_scales": [-1.0,-0.5,0.0,0.5],
        "train_feats": False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__()
        self.backbone = VGGBNPretrain()
        if self.cfg["train_feats"]:
            # disable BN training only
            disable_bn_grad(self.backbone)
        else:
            for para in self.backbone.parameters():
                para.requires_grad = False

        self.pool_ratio = 8
        self.img_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        d = 64
        self.score_conv=nn.Sequential(
            nn.Conv3d(3*len(self.cfg['detection_scales']),d,1,1),
            nn.ReLU(),
            nn.Conv3d(d,d,1,1),
        )
        self.score_predict=nn.Sequential(
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,1,3,1,1),
        )
        self.scale_predict=nn.Sequential(
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,1,3,1,1),
        )
        self.offset_predict=nn.Sequential(
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,d,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d,2,3,1,1),
        )
        self.ref_center_feats=None
        self.ref_shape=None

    def extract_feats(self, imgs):
        imgs = self.img_norm(imgs)
        if self.cfg['train_feats']:
            disable_bn_track(self.backbone)
            x0, x1, x2 = self.backbone(imgs)
        else:
            self.backbone.eval()
            with torch.no_grad():
                x0, x1, x2 = self.backbone(imgs)
        return x0, x1, x2

    def load_impl(self, ref_imgs):
        # resize to 120,120
        ref_imgs = F.interpolate(ref_imgs,size=(120,120))
        # 15, 7, 3
        self.ref_imgs = ref_imgs
        self.ref_center_feats = self.extract_feats(ref_imgs)
        rfn, _, h, w = ref_imgs.shape
        self.ref_shape = [h, w]

    def normalize_scores(self,scores0,scores1,scores2):
        stats = self.cfg['vgg_score_stats']
        scores0 = (scores0 - stats[0][0])/stats[0][1]
        scores1 = (scores1 - stats[1][0])/stats[1][1]
        scores2 = (scores2 - stats[2][0])/stats[2][1]

        scores0 = torch.clip(scores0,max=self.cfg['vgg_score_max'],min=-self.cfg['vgg_score_max'])
        scores1 = torch.clip(scores1,max=self.cfg['vgg_score_max'],min=-self.cfg['vgg_score_max'])
        scores2 = torch.clip(scores2,max=self.cfg['vgg_score_max'],min=-self.cfg['vgg_score_max'])
        return scores0, scores1, scores2

    def get_scores(self, que_imgs):
        que_x0, que_x1, que_x2 = self.extract_feats(que_imgs)
        ref_x0, ref_x1, ref_x2 = self.ref_center_feats # rfn,f,hr,wr

        scores2 = F.conv2d(que_x2, ref_x2, padding=1)
        scores1 = F.conv2d(que_x1, ref_x1, padding=3)
        scores0 = F.conv2d(que_x0, ref_x0, padding=7)
        scores2 = F.interpolate(scores2, scale_factor=4)
        scores1 = F.interpolate(scores1, scale_factor=2)
        scores0, scores1, scores2 = self.normalize_scores(scores0, scores1, scores2)

        scores = torch.stack([scores0, scores1, scores2],1) # qn,3,rfn,hq/8,wq/8
        return scores

    def detect_impl(self, que_imgs):
        qn, _, hq, wq = que_imgs.shape
        print(que_imgs.shape)
        hs, ws = hq // 8, wq // 8
        scores = []
        for scale in self.cfg['detection_scales']:
            ht, wt = int(np.round(hq*2**scale)), int(np.round(wq*2**scale))
            if ht%32!=0: ht=(ht//32+1)*32
            if wt%32!=0: wt=(wt//32+1)*32
            que_imgs_cur = F.interpolate(que_imgs,size=(ht,wt),mode='bilinear')
            scores_cur = self.get_scores(que_imgs_cur)
            qn, _, rfn, hcs, wcs = scores_cur.shape
            scores.append(F.interpolate(scores_cur.reshape(qn,3*rfn,hcs,wcs),size=(hs,ws),mode='bilinear').reshape(qn,3,rfn,hs,ws))

        scores = torch.cat(scores, 1) # qn,sn*3,rfn,hq/8,wq/8
        scores = self.score_conv(scores)
        scores_feats = torch.max(scores,2)[0] # qn,f,hq/8,wq/8
        scores = self.score_predict(scores_feats) # qn,1,hq/8,wq/8

        # predict offset and bbox
        _, select_w_id, select_h_id = self.get_select_index(scores)
        que_select_id = torch.stack([select_w_id, select_h_id],1) # qn, 2

        select_offset = self.offset_predict(scores_feats)  # qn,1,hq/8,wq/8
        select_scale = self.scale_predict(scores_feats) # qn,1,hq/8,wq/8
        outputs = {'scores': scores, 'que_select_id': que_select_id, 'pool_ratio': self.pool_ratio, 'select_pr_offset': select_offset, 'select_pr_scale': select_scale,}

        # bboxes_pr = []
        # for qi in range(que_imgs.shape[0]):
        #     bboxes_pr.append(parse_bbox_from_scale_offset(
        #         que_select_id[qi].detach().cpu().numpy(), select_scale[qi,0].detach().cpu().numpy(),
        #         select_offset[qi].detach().cpu().numpy(), self.pool_ratio, self.ref_shape))
        # outputs['bboxes_pr'] = np.stack(bboxes_pr,0)

        # decode bbox
        return outputs

    def forward(self, data):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()

        ref_imgs = ref_imgs_info['imgs']
        self.load_impl(ref_imgs)
        outputs = self.detect_impl(que_imgs_info['imgs'])
        return outputs

    def load_ref_imgs(self, ref_imgs):
        """
        @param ref_imgs: [an,rfn,h,w,3] in numpy
        @return:
        """
        # ref_imgs = torch.from_numpy(color_map_forward(ref_imgs)).permute(0,1,4,2,3) # an,rfn,3,h,w
        # ref_imgs = ref_imgs.cuda()
        # an,rfn,_,h,w = ref_imgs.shape
        # self.load_impl(ref_imgs[an//2])
        ref_imgs = torch.from_numpy(color_map_forward(ref_imgs)).permute(0,3,1,2) # rfn,3,h,w
        ref_imgs = ref_imgs.cuda()
        rfn, _, h, w = ref_imgs.shape
        self.load_impl(ref_imgs)

    def detect_que_imgs(self, que_img):
        """
        @param que_imgs: [h,w,3]
        @return:
        """
        que_imgs = torch.from_numpy(color_map_forward(que_img[None])).permute(0,3,1,2).cuda()
        qn, _, h, w = que_imgs.shape
        outputs = self.detect_impl(que_imgs)
        positions, scales = self.parse_detection(
            outputs['scores'].detach(), outputs['select_pr_scale'].detach(),
            outputs['select_pr_offset'].detach(), self.pool_ratio)
        detection_result = {'positions': positions, 'scales': scales}
        detection_result = to_cpu_numpy(detection_result)
        return [detection_result]

# 引入yoloE模块进行预处理
from ultralytics import YOLOE
class MultDetector(Detector):
    default_cfg={
        "vgg_score_stats": [[36.264317,13.151907],[13910.291,5345.965],[829.70807,387.98788]],
        "vgg_score_max": 10,
        "detection_scales": [-1.0,-0.5,0.0,0.5],
        "train_feats": False,
        "yolo_model": './data/model/yolo/yoloe-11l-seg.pt'
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__(self.cfg)
        self.yolo_model = None
    
    def load_yolo(self):
        print('load yolo model from',self.cfg['yolo_model'])
        self.yolo_model = YOLOE(self.cfg['yolo_model'])
        self.generate_ref_label()
        
    def generate_ref_label(self,prompt='chair'):
        names = [prompt]
        self.yolo_model.set_classes(names, self.yolo_model.get_text_pe(names))

    def yoloe_detect(self, img:Image):
        '''
        img: PIL:Image
        返回单张图检测框的中心
        '''
        
        # 使用翻转后的图像进行检测
        img_result = self.yolo_model.predict(img)[0]
        
        # imgs_centers = []
        # for img_idx, img_result in enumerate(results):
        obj_bboxes = []
        # 只有一类检测
        for bboxes in img_result.boxes:
            for bbox in bboxes.xyxyn:
                x1, y1, x2, y2 = bbox.tolist()
                obj_bboxes.append([x1, y1, x2, y2])
        return obj_bboxes
    
    def load_impl(self, ref_imgs):
        self.visual_prompt = dict(
            bboxes=[
                np.array(
                    [
                        [0.0, 120.0, 0.0, 120.0],  # Box enclosing person
                    ]
                ),
            ],
            cls=[
                np.array(
                    [
                        0,  # ID to be assigned for target
                    ]
                ),
                np.array([0]),
            ],
        )



    def extract_object(self, que_img, bboxes):
        """
        输入一张图像和 bbox 列表，返回提取的物体图像（bbox 外像素设为 0）。
        
        Args:
            que_img: 输入图像（NumPy 数组，H×W×C）。
            bboxes: 边界框列表，每个 bbox 格式为 [x1, y1, x2, y2]（相对坐标，0-1）。
        
        Returns:
            List[np.ndarray]: 提取的物体图像列表（每个物体对应一个图像）。
        """
        extracted_objects = []
        h, w = que_img.shape[:2]  # 获取图像高度和宽度
        img_bboxes = []
        for bbox in bboxes:
            print(bbox)
            # 将相对坐标转换为绝对坐标
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = h - int(y1 * h), h - int(y2 * h)
            
            # 确保坐标在图像范围内
            x1, x2 = sorted([max(0, x1), min(w, x2)])
            y1, y2 = sorted([max(0, y1), min(h, y2)])
            # 创建全零掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            img_bboxes.append([x1,y1,x2,y2])
            # 在bbox区域内设置为1
            mask[y1:y2, x1:x2] = 1
            # 打印mask区域大小
            # 应用掩码提取物体
            if len(que_img.shape) == 3:  # 彩色图像
                obj = que_img * mask[:, :, np.newaxis]
            else:  # 灰度图像
                obj = que_img * mask
            # if transpose:
            # obj = np.array(Image.fromarray(obj).transpose(Image.FLIP_TOP_BOTTOM))
            extracted_objects.append(obj)
        return extracted_objects,img_bboxes
    
    def detect_que_imgs(self, que_img):
        """
        @param que_imgs: [h,w,3]
        @return:
        """
        _, h, w = que_img.shape
        img = Image.fromarray(que_img)
        # 上下翻转
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save('./que_img_yolo.jpg')
        img = np.array(img)
        # 使用yolo先检测多个物体的位置，然后将bbox外的像素设置为空，给原模型检测
        normal_bboxes = self.yoloe_detect(img)
        object_imgs,img_bboxes = self.extract_object(que_img,normal_bboxes)
        # 保存que_imgs用于debug
        for i,img in enumerate(object_imgs):
            imsave(f'./que_imgs_{i}.png',img)
        # 打印normal_bboxes的 center 点和 bbox 大小和 img 尺寸关系

        detection_results = []
        for new_img,img_bbox in zip(object_imgs,img_bboxes):
            bbox_w,bbox_h = abs(img_bbox[2]-img_bbox[0])/2,abs(img_bbox[3]-img_bbox[1])
            scales = [max(bbox_w,bbox_h)/128]
            new_img =  color_map_forward(new_img)
            new_img = torch.from_numpy(new_img[None]).permute(0,3,1,2).cuda()
            positions = np.array([[(img_bbox[2]+img_bbox[0])/2,(img_bbox[3]+img_bbox[1])/2]])
            # outputs = self.detect_impl(new_img)
            # positions, scales = self.parse_detection(
            #     outputs['scores'].detach(), outputs['select_pr_scale'].detach(),
            #     outputs['select_pr_offset'].detach(), self.pool_ratio)
            detection_result = {'positions': positions, 'scales': scales}
            detection_result = to_cpu_numpy(detection_result)
            detection_results.append(detection_result)
        torch.save([normal_bboxes,detection_results],'tmp.torch')
        return detection_results