import argparse
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.io import imsave, imread
from tqdm import tqdm
import sys,os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.database import parse_database_name, get_ref_point_cloud,GSDatabase
from estimator import name2estimator
from eval import visualize_intermediate_results
from prepare import video2image
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp
import shutil


def main(args):
    cfg = load_cfg(args.cfg)
    ref_database = parse_database_name(args.database)
    estimator = name2estimator[cfg['type']](cfg)
    estimator.build(ref_database, split_type='all')

    object_pts = get_ref_point_cloud(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    subdirs = [ 'images_out', 'images_inter']

    for subdir in subdirs:
        dir_path = output_dir / subdir
        # 如果目录存在，则删除
        if dir_path.exists():
            shutil.rmtree(dir_path)
        # 重新创建目录
        dir_path.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(Path(args.video_path).glob('*.jpg')) 
    # pose_init = None
    for que_id,img_path in tqdm(enumerate(img_paths)):
        stem_name = img_path.stem
        img = imread(img_path)
        # generate a pseudo K
        h, w, _ = img.shape
        f = np.sqrt(h**2+w**2)
        K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)

        # if pose_init is not None:
        # estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
        pose_prs, inter_results = estimator.predict(img, K)
        if len(pose_prs) == 0:
            continue
        bbox_img = None
        for i,(pose_pr,inter_result) in enumerate(zip(pose_prs,inter_results)):
            pts, _ = project_points(object_bbox_3d, pose_pr, K)
            if bbox_img is None:
                bbox_img = draw_bbox_3d(img, pts, (0,0,255))
            else:
                bbox_img = draw_bbox_3d(bbox_img, pts, (0,0,255))
            imsave(f'{str(output_dir)}/images_inter/{stem_name}_{i}.jpg', visualize_intermediate_results(img, K, inter_result, estimator.ref_info, object_bbox_3d))

        imsave(f'{str(output_dir)}/images_out/{stem_name}-bbox.jpg', bbox_img)
        np.save(f'{str(output_dir)}/images_out/{stem_name}-pose.npy', pose_prs)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
    parser.add_argument('--database', type=str, default="custom/mouse")
    parser.add_argument('--output', type=str, default="data/custom/mouse/test")
    parser.add_argument('--video_path', type=str, default="/workspace/qian.ren/shangfei_codes/0820/Gen6D/data/gs/chair/test/images·")
    args = parser.parse_args()
    main(args)
