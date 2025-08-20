import abc
import glob
from pathlib import Path
import sys
sys.path.append('gaussian-splatting')
import cv2
import numpy as np
import os
import torch
import plyfile
from PIL import Image
from skimage.io import imread, imsave

from utils.base_utils import read_pickle, save_pickle, pose_compose, load_point_cloud, pose_inverse, resize_img, \
    mask_depth_to_pts, transform_points_pose
from utils.read_write_model import read_model

SUN_IMAGE_ROOT = 'data/SUN2012pascalformat/JPEGImages'
SUN_IMAGE_ROOT_128 = 'data/SUN2012pascalformat/JPEGImages_128'
SUN_IMAGE_ROOT_256 = 'data/SUN2012pascalformat/JPEGImages_256'
SUN_IMAGE_ROOT_512 = 'data/SUN2012pascalformat/JPEGImages_512'
SUN_IMAGE_ROOT_32 = 'data/SUN2012pascalformat/JPEGImages_64'
def get_SUN397_image_fn_list():
    if Path('data/SUN397_list.pkl').exists():
        return read_pickle('data/SUN397_list.pkl')
    img_list = os.listdir(SUN_IMAGE_ROOT)
    img_list = [img for img in img_list if img.endswith('.jpg')]
    save_pickle(img_list, 'data/SUN397_list.pkl')
    return img_list

class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self):
        pass

    def get_mask(self,img_id):
        # dummy mask
        img = self.get_image(img_id)
        h, w = img.shape[:2]
        return np.ones([h,w],bool)

LINEMOD_ROOT='data/LINEMOD'
class LINEMODDatabase(BaseDatabase):
    K=np.array([[572.4114, 0., 325.2611],
               [0., 573.57043, 242.04899],
               [0., 0., 1.]], dtype=np.float32)
    def __init__(self, database_name):
        super().__init__(database_name)
        _, self.model_name = database_name.split('/')
        self.img_ids = [str(k) for k in range(len(os.listdir(f'{LINEMOD_ROOT}/{self.model_name}/JPEGImages')))]
        self.model = self.get_ply_model().astype(np.float32)
        self.object_center = np.zeros(3,dtype=np.float32)
        self.object_vert = np.asarray([0,0,1],np.float32)
        self.img_id2depth_range = {}
        self.img_id2pose = {}

    def get_ply_model(self):
        fn = Path(f'{LINEMOD_ROOT}/{self.model_name}/{self.model_name}.pkl')
        if fn.exists(): return read_pickle(str(fn))
        ply = plyfile.PlyData.read(f'{LINEMOD_ROOT}/{self.model_name}/{self.model_name}.ply')
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        if model.shape[0]>4096:
            idxs = np.arange(model.shape[0])
            np.random.shuffle(idxs)
            model = model[idxs[:4096]]
        save_pickle(model, str(fn))
        return model

    def get_image(self, img_id):
        return imread(f'{LINEMOD_ROOT}/{self.model_name}/JPEGImages/{int(img_id):06}.jpg')

    def get_K(self, img_id):
        return np.copy(self.K)

    def get_pose(self, img_id):
        if img_id in self.img_id2pose:
            return self.img_id2pose[img_id]
        else:
            pose = np.load(f'{LINEMOD_ROOT}/{self.model_name}/pose/pose{int(img_id)}.npy')
            self.img_id2pose[img_id] = pose
            return pose

    def get_img_ids(self):
        return self.img_ids.copy()

    def get_mask(self, img_id):
        return np.sum(imread(f'{LINEMOD_ROOT}/{self.model_name}/mask/{int(img_id):04}.png'),-1)>0

GenMOP_ROOT='data/GenMOP'

genmop_meta_info={
    'cup': {'gravity': np.asarray([-0.0893124,-0.399691,-0.912288]), 'forward': np.asarray([-0.009871,0.693020,-0.308549],np.float32)},
    'tformer': {'gravity': np.asarray([-0.0734401,-0.633415,-0.77032]), 'forward': np.asarray([-0.121561, -0.249061, 0.211048],np.float32)},
    'chair': {'gravity': np.asarray((0.111445, -0.373825, -0.920779),np.float32), 'forward': np.asarray([0.788313,-0.139603,0.156288],np.float32)},
    'knife': {'gravity': np.asarray((-0.0768299, -0.257446, -0.963234),np.float32), 'forward': np.asarray([0.954157,0.401808,-0.285027],np.float32)},
    'love': {'gravity': np.asarray((0.131457, -0.328559, -0.93529),np.float32), 'forward': np.asarray([-0.045739,-1.437427,0.497225],np.float32)},
    'plug_cn': {'gravity': np.asarray((-0.0267497, -0.406514, -0.913253),np.float32), 'forward': np.asarray([-0.172773,-0.441210,0.216283],np.float32)},
    'plug_en': {'gravity': np.asarray((0.0668682, -0.296538, -0.952677),np.float32), 'forward': np.asarray([0.229183,-0.923874,0.296636],np.float32)},
    'miffy': {'gravity': np.asarray((-0.153506, -0.35346, -0.922769),np.float32), 'forward': np.asarray([-0.584448,-1.111544,0.490026],np.float32)},
    'scissors': {'gravity': np.asarray((-0.129767, -0.433414, -0.891803),np.float32), 'forward': np.asarray([1.899760,0.418542,-0.473156],np.float32)},
    'piggy': {'gravity': np.asarray((-0.122392, -0.344009, -0.930955), np.float32), 'forward': np.asarray([0.079012,1.441836,-0.524981], np.float32)},
}
class GenMOPMetaInfoWrapper:
    def __init__(self, object_name):
        self.object_name = object_name
        self.gravity = genmop_meta_info[self.object_name]['gravity']
        self.forward = genmop_meta_info[self.object_name]['forward']
        self.object_point_cloud = load_point_cloud(f'{GenMOP_ROOT}/{self.object_name}-ref/object_point_cloud.ply')

        # rotate
        self.rotation = self.compute_rotation(self.gravity, self.forward)
        self.object_point_cloud = (self.object_point_cloud @ self.rotation.T)

        # scale
        self.scale_ratio = self.compute_normalized_ratio(self.object_point_cloud)
        self.object_point_cloud = self.object_point_cloud * self.scale_ratio

        min_pt = np.min(self.object_point_cloud,0)
        max_pt = np.max(self.object_point_cloud,0)
        self.center = (max_pt + min_pt)/2

        test_fn = f'{GenMOP_ROOT}/{self.object_name}-ref/test-object_point_cloud.ply'
        if Path(test_fn).exists():
            self.test_object_point_cloud = load_point_cloud(test_fn)

    @staticmethod
    def compute_normalized_ratio(pc):
        min_pt = np.min(pc,0)
        max_pt = np.max(pc,0)
        dist = np.linalg.norm(max_pt - min_pt)
        scale_ratio = 2.0 / dist
        return scale_ratio

    def normalize_pose(self, pose):
        R = pose[:3,:3]
        t = pose[:3,3:]
        R = R @ self.rotation.T
        t = self.scale_ratio * t
        return np.concatenate([R,t], 1).astype(np.float32)

    @staticmethod
    def compute_rotation(vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert/np.linalg.norm(vert)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R

def parse_colmap_project(cameras, images, img_fns):
    """解析COLMAP项目，提取相机位姿和内参"""
    # 获取第一个图像来判断是否为Windows COLMAP格式
    v = images[list(images.keys())[0]]
    is_windows_colmap = v.name.startswith('frame')
    
    # 建立图像名称到数据库ID的映射
    if is_windows_colmap:
        img_id2db_id = {v.name: k for k, v in images.items()}
    else:
        img_id2db_id = {v.name[:-4]: k for k, v in images.items()}
    
    poses, Ks = {}, {}
    img_ids = [str(k) for k in range(len(img_fns))]
    print(img_ids)
    # 遍历所有图像ID，提取位姿和内参
    for img_id in img_ids:
        if is_windows_colmap:
            if img_fns[int(img_id)] not in img_id2db_id: 
                continue
            db_id = img_id2db_id[img_fns[int(img_id)]]
        else:
            if img_id not in img_id2db_id: 
                continue
            db_id = img_id2db_id[img_id]
        
        # 提取旋转矩阵和平移向量
        R = images[db_id].qvec2rotmat()  # 将四元数转换为旋转矩阵
        t = images[db_id].tvec
        pose = np.concatenate([R, t[:, None]], 1).astype(np.float32)
        poses[img_id] = pose

        # 提取相机内参
        cam_id = images[db_id].camera_id
        camera = cameras[cam_id]
        
        # 根据相机模型提取内参
        if camera.model == 'PINHOLE':
            fx, fy, cx, cy = camera.params
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        elif camera.model == 'SIMPLE_PINHOLE':
            f, cx, cy = camera.params
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
        elif camera.model == "SIMPLE_RADIAL":
            # 对于其他相机模型，假设前4个参数是 fx, fy, cx, cy
            f,  cx, cy, _ = camera.params[:4]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
        else:
            raise NotImplementedError

        
        Ks[img_id] = K
    return poses, Ks, img_ids

class GenMOPDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)

        _, seq_name = database_name.split('/') # genmop/object_name-test or genmop/object_name-ref

        # get image filenames
        self.seq_name = seq_name
        self.root = Path(GenMOP_ROOT) / self.seq_name
        img_fns_cache = self.root / 'images_fn_cache.pkl'
        self.img_fns = read_pickle(str(img_fns_cache))

        # parse colmap project
        cameras, images, points3d = read_model(f'{GenMOP_ROOT}/{seq_name}/colmap-all/colmap_default-colmap_default/sparse/0')
        self.poses, self.Ks, self.img_ids = parse_colmap_project(cameras, images, self.img_fns)

        # align test sequence to the reference sequence
        object_name, database_type = seq_name.split('-')
        if database_type=='test':
            scale_ratio, transfer_pose = read_pickle(f'{GenMOP_ROOT}/{seq_name}/align.pkl')
            for img_id in self.get_img_ids():
                pose = self.poses[img_id]
                pose_new = pose_compose(transfer_pose, pose)
                pose_new[:, 3:] *= scale_ratio
                self.poses[img_id] = pose_new

        # normalize object poses by meta info: rotate and scale not offset
        self.meta_info = GenMOPMetaInfoWrapper(object_name)
        self.poses = {img_id: self.meta_info.normalize_pose(self.poses[img_id]) for img_id in self.get_img_ids()}

    def get_image(self, img_id):
        return imread(str(self.root / 'images' / self.img_fns[int(img_id)]))

    def get_K(self, img_id):
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

class CustomDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        self.root = Path(os.path.join('data',database_name))
        self.img_dir = self.root / 'images'
        if (self.root/'img_fns.pkl').exists():
            self.img_fns = read_pickle(str(self.root/'img_fns.pkl'))
        else:
            self.img_fns = [Path(fn).name for fn in glob.glob(str(self.img_dir)+'/*.jpg')]
            save_pickle(self.img_fns, str(self.root/'img_fns.pkl'))

        self.colmap_root = self.root / 'colmap'
        if (self.colmap_root / 'sparse' / '0').exists():
            cameras, images, points3d = read_model(str(self.colmap_root / 'sparse' / '0'))
            self.poses, self.Ks, self.img_ids = parse_colmap_project(cameras, images, self.img_fns)
        else:
            self.img_ids = [str(k) for k in range(len(self.img_fns))]
            self.poses, self.Ks = {}, {}

        if len(self.poses.keys())>0:
            # read meta information to scale and rotate
            directions = np.loadtxt(str(self.root/'meta_info.txt'))
            x = directions[0]
            z = directions[1]
            self.object_point_cloud = load_point_cloud(f'{self.root}/object_point_cloud.ply')
            # rotate
            self.rotation = GenMOPMetaInfoWrapper.compute_rotation(z, x)
            self.object_point_cloud = (self.object_point_cloud @ self.rotation.T)

            # scale
            self.scale_ratio = GenMOPMetaInfoWrapper.compute_normalized_ratio(self.object_point_cloud)
            self.object_point_cloud = self.object_point_cloud * self.scale_ratio
            min_pt = np.min(self.object_point_cloud, 0)
            max_pt = np.max(self.object_point_cloud, 0)
            self.center = (max_pt + min_pt) / 2

            # 修改位姿
            for k, pose in self.poses.items():
                R = pose[:3, :3]
                t = pose[:3, 3:]
                R = R @ self.rotation.T
                t = self.scale_ratio * t
                self.poses[k] = np.concatenate([R,t], 1).astype(np.float32)

    def get_image(self, img_id):
        """获取指定ID的图像"""
        return imread(str(self.img_dir/self.img_fns[int(img_id)]))

    def get_K(self, img_id):
        """获取相机内参矩阵"""
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        """获取相机位姿"""
        return self.poses[img_id].copy()

    def get_img_ids(self):
        """获取所有图像ID"""
        return self.img_ids

class GSDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        # 设置数据根目录
        self.root = Path(os.path.join('data',database_name))
        self.img_dir = self.root / 'colmap' / 'images'
        # 获取图像文件名列表
        if (self.root/'img_fns.pkl').exists():
            self.img_fns = read_pickle(str(self.root/'img_fns.pkl'))
        else:
            self.img_fns = [Path(fn).name for fn in glob.glob(str(self.img_dir)+'/*.jpg')]
            save_pickle(self.img_fns, str(self.root/'img_fns.pkl'))
        
        # 解析COLMAP项目（如果存在）
        self.colmap_root = self.root / 'colmap' 
        if (self.colmap_root / 'sparse' / '0').exists():
            cameras, images, points3d = read_model(str(self.colmap_root / 'sparse' / '0'))
            self.poses, self.Ks, self.img_ids = parse_colmap_project(cameras, images, self.img_fns)
        else:
            raise ValueError('没有加载位姿信息')
        # 获取默认图像
        self.image_size = Image.open(str(self.img_dir/self.img_fns[0])).size
        # image_width, image_height = image_size
        # 初始化3D Gaussian Splatting相关参数
        self.gaussian_model = None
        self.renderer = None
    
        # 读取元信息进行缩放和旋转
        directions = np.loadtxt(str(self.root/'meta_info.txt'))
        x = directions[0]  # 前向方向
        z = directions[1]  # 垂直方向
        
        # 初始化3D Gaussian Splatting模型（不进行变换）
        self._initialize_3dgs_model()
        
        # 计算旋转矩阵,并变换
        self.rotation = GenMOPMetaInfoWrapper.compute_rotation(z, x)
        self.gaussian_model.rotate_by_matrix(self.rotation)

        # 从3DGS模型获取点云用于缩放计算
        # self.scale_ratio = 1
        self.scale_ratio = self.compute_normalized_ratio(self.gaussian_model.get_xyz)
        print('scale_ratio',self.scale_ratio)
        self.gaussian_model.rescale(self.scale_ratio)
        gaussian_points = self.gaussian_model.get_xyz

        # 计算物体中心（基于原始高斯点）
        min_pt = torch.min(gaussian_points, dim=0)[0]  # torch.min 返回 (values, indices)，取 values
        max_pt = torch.max(gaussian_points, dim=0)[0]  # torch.max 返回 (values, indices)，取 values
        self.center = ((max_pt + min_pt) / 2).detach().cpu().numpy()  # 转换为numpy用于后续计算
        print('Center',self.center)
        # 修改位姿以匹配变换参数
        for k, pose in self.poses.items():
            R = pose[:3, :3]
            t = pose[:3, 3:]
            R = R @ self.rotation.T
            t = self.scale_ratio.detach().cpu().numpy() * t
            self.poses[k] = np.concatenate([R,t], 1).astype(np.float32)
            
    
    def _initialize_3dgs_model(self):
        """初始化3D Gaussian Splatting模型"""
        # 导入3DGS相关模块（需要根据实际使用的3DGS库调整）

        from scene import Scene
        from gaussian_renderer import render
        from utils.camera_utils import Camera
        
        # 加载预训练的3DGS模型或从点云初始化
        gaussian_model_path = self.root / '3dgs' / 'point_cloud' / 'iteration_30000' / 'chair.ply'
        print("Loading 3DGS model from:", gaussian_model_path)
        # 从保存的高斯模型加载
        self.gaussian_model = self._load_gaussian_model(str(gaussian_model_path))
        # 初始化 pipeline 参数
        class SimplePipelineParams:
            def __init__(self):
                self.convert_SHs_python = False
                self.compute_cov3D_python = False
                self.debug = False
                self.antialiasing = False
        
        self.pipeline = SimplePipelineParams()
        
        # 初始化渲染器
        self.renderer = render
            
    
    def _load_gaussian_model(self, model_path):
        """从文件加载3D高斯模型"""
        # 这里需要根据具体的3DGS实现来加载模型
        # 示例代码，需要根据实际库调整
        import sys
        sys.path.append('gaussian-splatting')
        from scene.gaussian_model import GaussianModel
        model = GaussianModel(sh_degree=3)
        model.load_ply(model_path)
        return model

    def _get_gaussian_centers(self):
        """获取高斯中心点坐标"""

        centers = self.gaussian_model.get_xyz.detach().cpu().numpy()
        return centers
    
    
    def render(self, pose, K,  width=None,height=None,background_color=None):
        """使用3D Gaussian Splatting进行新视角合成渲染
        
        Args:
            pose: 相机位姿矩阵 [3x4] 或 [4x4]
            K: 相机内参矩阵 [3x3]
            image_height: 渲染图像高度
            image_width: 渲染图像宽度
            background_color: 背景颜色，默认为白色
            
        Returns:
            rendered_image: 渲染的图像 [H, W, 3]
            rendered_depth: 渲染的深度图 [H, W] (可选)
        """
        if self.gaussian_model is None or self.renderer is None:
            raise RuntimeError("3D Gaussian Splatting模型未初始化，无法进行渲染")
        if width  and height:
            image_size = (width, height)
        else:
            image_size = self.image_size

        import torch
        from utils.camera_utils import Camera
        
        # 处理位姿矩阵格式
        if pose.shape == (3, 4):
            # 转换为4x4矩阵
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :] = pose
        else:
            pose_4x4 = pose.astype(np.float32)
        R = pose_4x4[:3, :3]
        t = pose_4x4[:3, 3]
        
        # 创建相机对象
        camera = Camera(
            colmap_id=0,
            R=R.T,
            T=t,
            depth_params= None,
            invdepthmap = None,
            resolution = self.image_size ,
            FoVx=2 * np.arctan(image_size[0] / (2 * K[0, 0])),
            FoVy=2 * np.arctan(image_size[1] / (2 * K[1, 1])),
            image=Image.new('RGB', image_size, (0, 0, 0)),
            image_name="rendered",
            uid=0,
            data_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 设置背景颜色
        if background_color is None:
            background_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            background_color = torch.tensor(background_color, dtype=torch.float32)
        
        if torch.cuda.is_available():
            background_color = background_color.cuda()
        
        # 执行渲染
        with torch.no_grad():
            render_pkg = self.renderer(camera, self.gaussian_model,self.pipeline, background_color)
            rendered_image = render_pkg["render"]
            
            # 转换为numpy格式 [H, W, 3]
            rendered_image = rendered_image.permute(1, 2, 0).cpu().numpy()
            rendered_image = np.clip(rendered_image, 0, 1)
            
            # 可选：返回深度图
            rendered_depth = None
            if "depth" in render_pkg:
                rendered_depth = render_pkg["depth"].cpu().numpy()
        # conver to np.uint8
        rendered_image = (rendered_image * 255).astype(np.uint8)
        return rendered_image, rendered_depth
    @property
    def object_point_cloud(self):
        return self.gaussian_model.get_xyz.detach().cpu().numpy()

    def get_image(self, img_id,origin=False):
        """获取指定ID的图像"""
        if origin:
            return imread(str(self.img_dir/self.img_fns[int(img_id)]))
        K = self.get_K(img_id)
        pose = self.get_pose(img_id)
        return self.render(pose, K)[0]

    def get_K(self, img_id):
        """获取相机内参矩阵"""
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        """获取相机位姿"""
        return self.poses[img_id].copy()

    def get_img_ids(self):
        """获取所有图像ID"""
        return self.img_ids
    
    def save_gaussian_model(self, save_path=None):
        """保存3D高斯模型到文件"""
        if self.gaussian_model is None:
            print("警告: 没有可保存的3D高斯模型")
            return False
        
        if save_path is None:
            save_path = self.root / 'gaussian_model.ply'
        
        try:
            self.gaussian_model.save_ply(str(save_path))
            print(f"3D高斯模型已保存到: {save_path}")
            return True
        except Exception as e:
            print(f"保存3D高斯模型失败: {e}")
            return False
    
    def get_gaussian_properties(self):
        """获取3D高斯模型的属性信息"""
        properties = {
            'num_gaussians': self.gaussian_model.get_xyz.shape[0],
            'centers': self.gaussian_model.get_xyz.detach().cpu().numpy(),
            'scales': self.gaussian_model.get_scaling.detach().cpu().numpy(),
            'rotations': self.gaussian_model.get_rotation.detach().cpu().numpy(),
            'opacities': self.gaussian_model.get_opacity.detach().cpu().numpy(),
            'colors': self.gaussian_model.get_features.detach().cpu().numpy()
        }
        return properties


    def compute_normalized_ratio(self,pc, device=None):
        """计算归一化比例，使物体直径为2"""
        # 确保输入是 torch tensor 并在正确设备上
        if not isinstance(pc, torch.Tensor):
            pc = torch.tensor(pc, dtype=torch.float32)
        
        if device is not None:
            pc = pc.to(device)
        
        min_pt = torch.min(pc, dim=0)[0]
        max_pt = torch.max(pc, dim=0)[0]
        dist = torch.norm(max_pt - min_pt)
        scale_ratio = 2.0 / dist
        return scale_ratio.detach()

def parse_database_name(database_name:str)->BaseDatabase:
    name2database={
        'linemod': LINEMODDatabase,
        'genmop': GenMOPDatabase,
        'custom': CustomDatabase,
        'gs':GSDatabase,
        'co3d_resize': Co3DResizeDatabase,
        'shapenet': ShapeNetRenderDatabase,
        'gso': GoogleScannedObjectDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError

def get_database_split(database, split_name):
    if split_name.startswith('linemod'): # linemod_test or linemod_val
        assert(database.database_name.startswith('linemod'))
        model_name = database.database_name.split('/')[1]
        lines = np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/test.txt",dtype=np.str).tolist()
        que_ids, ref_ids = [], []
        for line in lines: que_ids.append(str(int(line.split('/')[-1].split('.')[0])))
        if split_name=='linemod_val': que_ids = que_ids[::10]
        lines = np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/train.txt", dtype=np.str).tolist()
        for line in lines: ref_ids.append(str(int(line.split('/')[-1].split('.')[0])))
    elif split_name=='all':
        ref_ids = que_ids = database.get_img_ids()
    else:
        raise NotImplementedError
    return ref_ids, que_ids

def get_ref_point_cloud(database):
    """获取参考点云"""
    if isinstance(database, LINEMODDatabase):
        ref_point_cloud = database.model
    elif isinstance(database, GenMOPDatabase):
        ref_point_cloud = database.meta_info.object_point_cloud
    elif isinstance(database, Co3DResizeDatabase) or isinstance(database, GoogleScannedObjectDatabase):
        raise NotImplementedError
    elif isinstance(database, ShapeNetRenderDatabase):
        return database.model_verts
    elif isinstance(database, CustomDatabase):
        ref_point_cloud = database.object_point_cloud
    elif isinstance(database, GSDatabase):
        ref_point_cloud = database.object_point_cloud
    elif isinstance(database, NormalizedDatabase):
        pc = get_ref_point_cloud(database.database)
        pc = pc * database.scale + database.offset
        return pc
    else:
        raise NotImplementedError
    return ref_point_cloud

def get_diameter(database):
    if isinstance(database, LINEMODDatabase):
        model_name = database.database_name.split('/')[-1]
        return np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/distance.txt") / 100
    elif isinstance(database, GenMOPDatabase):
        return 2.0 # we already align and scale it
    elif isinstance(database, GoogleScannedObjectDatabase):
        return database.object_diameter
    elif isinstance(database, Co3DResizeDatabase):
        raise NotImplementedError
    elif isinstance(database, ShapeNetRenderDatabase):
        return database.object_diameter
    elif isinstance(database, NormalizedDatabase):
        return 2.0
    elif isinstance(database, CustomDatabase):
        return 2.0
    elif isinstance(database, GSDatabase):
        return 2.0
    else:
        raise NotImplementedError

def get_object_center(database):
    if isinstance(database, LINEMODDatabase):
        return database.object_center
    elif isinstance(database, GenMOPDatabase):
        return database.meta_info.center
    elif isinstance(database, GoogleScannedObjectDatabase):
        return database.object_center
    elif isinstance(database, Co3DResizeDatabase):
        raise NotImplementedError
    elif isinstance(database, ShapeNetRenderDatabase):
        return database.object_center
    elif isinstance(database, CustomDatabase):
        return database.center
    elif isinstance(database, GSDatabase):
        return database.center
    elif isinstance(database, NormalizedDatabase):
        return np.zeros(3,dtype=np.float32)
    else:
        raise NotImplementedError

def get_object_vert(database):
    if isinstance(database, LINEMODDatabase):
        return database.object_vert
    elif isinstance(database, GenMOPDatabase):
        return np.asarray([0,0,1], np.float32)
    elif isinstance(database, GoogleScannedObjectDatabase):
        return database.object_vert
    elif isinstance(database, Co3DResizeDatabase):
        raise NotImplementedError
    elif isinstance(database, ShapeNetRenderDatabase):
        return database.object_vert
    elif isinstance(database, CustomDatabase):
        return np.asarray([0,0,1], np.float32)
    elif isinstance(database, GSDatabase):
        return np.asarray([0,0,1], np.float32)
    else:
        raise NotImplementedError

def normalize_pose(pose, scale, offset):
    # x_obj_new = x_obj * scale + offset
    R = pose[:3, :3]
    t = pose[:3, 3]
    t_ = R @ -offset + scale * t
    return np.concatenate([R, t_[:,None]], -1).astype(np.float32)

def denormalize_pose(pose, scale, offset):
    R = pose[:3,:3]
    t = pose[:3, 3]
    t = R @ offset / scale + t/scale
    return np.concatenate([R, t[:, None]], -1).astype(np.float32)

class GoogleScannedObjectDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, model_name, background_resolution = database_name.split('/')
        background, resolution = background_resolution.split('_')
        assert(background in ['black','white'])
        self.resolution = resolution
        self.background = background
        self.prefix=f'data/google_scanned_objects/{model_name}'

        if self.resolution!='raw':
            resolution = int(self.resolution)

            # cache images
            self.img_cache_prefix = f'data/google_scanned_objects/{model_name}/rgb_{resolution}'
            Path(self.img_cache_prefix).mkdir(exist_ok=True,parents=True)
            for img_id in self.get_img_ids():
                fn = Path(self.img_cache_prefix) / f'{int(img_id):06}.jpg'
                if fn.exists(): continue
                img = imread(f'{self.prefix}/rgb/{int(img_id):06}.png')[:, :, :3]
                img = resize_img(img, resolution / 512)
                imsave(str(fn), img)

            # cache masks
            self.mask_cache_prefix = f'data/google_scanned_objects/{model_name}/mask_{resolution}'
            Path(self.mask_cache_prefix).mkdir(exist_ok=True, parents=True)
            for img_id in self.get_img_ids():
                fn = Path(self.mask_cache_prefix) / f'{int(img_id):06}.png'
                if fn.exists(): continue
                mask = imread(f'{self.prefix}/mask/{int(img_id):06}.png')>0
                mask = mask.astype(np.uint8)
                mask = cv2.resize(mask, (resolution,resolution), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(fn), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        #################compute object center###################
        self.model_name = model_name
        object_center_fn = f'data/google_scanned_objects/{model_name}/object_center.pkl'
        if os.path.exists(object_center_fn):
            self.object_center = read_pickle(object_center_fn)
        else:
            obj_pts = self.get_object_points()
            max_pt, min_pt = np.max(obj_pts,0), np.min(obj_pts,0)
            self.object_center = (max_pt+min_pt)/2
            save_pickle(self.object_center, object_center_fn)
        self.img_id2pose={}

        ################compute object vertical direction############
        vert_dir_fn = f'data/google_scanned_objects/{model_name}/object_vert.pkl'
        if os.path.exists(vert_dir_fn):
            self.object_vert = read_pickle(vert_dir_fn)
        else:
            poses = [self.get_pose(img_id) for img_id in self.get_img_ids()]
            cam_pts = [pose_inverse(pose)[:3, 3] for pose in poses]
            cam_pts_diff = np.asarray(cam_pts) - self.object_center[None,]
            self.object_vert = np.mean(cam_pts_diff, 0)
            save_pickle(self.object_vert, vert_dir_fn)

        #################compute object diameter###################
        object_diameter_fn = f'data/google_scanned_objects/{model_name}/object_diameter.pkl'
        if os.path.exists(object_diameter_fn):
            self.object_diameter = read_pickle(object_diameter_fn)
        else:
            self.object_diameter = self._get_diameter()
            save_pickle(self.object_diameter, object_diameter_fn)

    def get_raw_depth(self, img_id):
        img = Image.open(f'{self.prefix}/depth/{int(img_id):06}.png')
        depth = np.asarray(img, dtype=np.float32) / 1000.0
        mask = imread(f'{self.prefix}/mask/{int(img_id):06}.png')>0
        depth[~mask] = 0
        return depth

    def get_object_points(self):
        fn = f'data/gso_cache/{self.model_name}-pts.pkl'
        if os.path.exists(fn): return read_pickle(fn)
        obj_pts = []
        for img_id in self.get_img_ids():
            pose = self.get_pose(img_id)
            mask = self.get_mask(img_id)
            K = self.get_K(img_id)
            pose_inv = pose_inverse(pose)
            depth = self.get_raw_depth(img_id)
            pts = mask_depth_to_pts(mask, depth, K)
            pts = transform_points_pose(pts, pose_inv)
            idx = np.arange(pts.shape[0])
            np.random.shuffle(idx)
            idx = idx[:1024]
            pts = pts[idx]
            obj_pts.append(pts)
        obj_pts = np.concatenate(obj_pts, 0)
        save_pickle(obj_pts, fn)
        return obj_pts

    def _get_diameter(self):
        obj_pts = self.get_object_points()
        max_pt, min_pt = np.max(obj_pts, 0), np.min(obj_pts, 0)
        return np.linalg.norm(max_pt - min_pt)

    def get_image(self, img_id, ref_mode=False):
        if self.resolution!='raw':
            img = imread(f'{self.img_cache_prefix}/{int(img_id):06}.jpg')[:,:,:3]
            if self.background == 'black':
                mask = self.get_mask(img_id)
                img[~mask]=0
        else:
            img = imread(f'{self.prefix}/rgb/{int(img_id):06}.png')[:,:,:3]
            if self.background=='black':
                mask = imread(f'{self.prefix}/mask/{int(img_id):06}.png')>0
                img[~mask] = 0
        return img

    def get_K(self, img_id):
        K=np.loadtxt(f'{self.prefix}/intrinsics/{int(img_id):06}.txt').reshape([4,4])[:3,:3]
        if self.resolution!='raw':
            ratio = int(self.resolution) / 512
            K = np.diag([ratio,ratio,1.0]) @ K
        return np.copy(K.astype(np.float32))

    def get_pose(self, img_id):
        if img_id in self.img_id2pose:
            return self.img_id2pose[img_id].copy()
        else:
            pose = np.loadtxt(f'{self.prefix}/pose/{int(img_id):06}.txt').reshape([4,4])[:3,:]
            R = pose[:3, :3].T
            t = R @ -pose[:3, 3:]
            pose = np.concatenate([R,t],-1)
            self.img_id2pose[img_id] = pose
            return np.copy(pose)

    def get_img_ids(self):
        return [str(img_id) for img_id in range(250)]

    def get_mask(self, img_id):
        if self.resolution!='raw':
            mask = imread(f'{self.mask_cache_prefix}/{int(img_id):06}.png')>0
        else:
            mask=imread(f'{self.prefix}/mask/{int(img_id):06}.png')>0
        return mask

Co3D_ROOT = 'data/co3d'

def mask2bbox(mask):
    """将掩码转换为边界框"""
    if np.sum(mask)==0:
        return np.asarray([0, 0, 0, 0],np.float32)
    ys, xs = np.nonzero(mask)
    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)
    return np.asarray([x_min, y_min, x_max - x_min, y_max - y_min], np.int32)

class Co3DResizeDatabase(BaseDatabase):
    def __init__(self, database_name):
        super(Co3DResizeDatabase, self).__init__(database_name)
        _, self.category, self.sequence, sizes = database_name.split('/')
        self.fg_size, self.bg_size = [int(item) for item in sizes.split('_')]
        self._build_resize_database()

    def _build_resize_database(self):
        annotation_fn = Path(f'{Co3D_ROOT}_{self.fg_size}_{self.bg_size}/{self.category}/{self.sequence}/info.pkl')
        root_dir = annotation_fn.parent
        self.image_root = (root_dir / 'images')
        self.mask_root = (root_dir / 'masks')
        if annotation_fn.exists():
            self.Ks, self.poses, self.img_ids, self.ratios = read_pickle(str(annotation_fn))
        else:
            raise NotImplementedError

    def get_image(self, img_id, ref_mode=False):
        return imread(str(self.image_root / f'{img_id}.jpg'))

    def get_K(self, img_id):
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

    def get_bbox(self, img_id):
        return mask2bbox(self.get_mask(img_id))

    def get_mask(self, img_id):
        return imread(str(self.mask_root / f'{img_id}.png')) > 0

SHAPENET_RENDER_ROOT='data/shapenet/shapenet_render'

class ShapeNetRenderDatabase(BaseDatabase):
    def __init__(self, database_name):
        super(ShapeNetRenderDatabase, self).__init__(database_name)
        # shapenet/02691156/1ba18539803c12aae75e6a02e772bcee/evenly-32-128
        _, self.category, self.model_name, self.render_setting = database_name.split('/')
        self.render_num = int(self.render_setting.split('-')[1])
        self.object_vert = np.asarray([0,1,0],np.float32)

        self.img_id2camera={}
        cache_fn=Path(f'data/shapenet/shapenet_cache/{self.category}-{self.model_name}-{self.render_setting}.pkl')
        if cache_fn.exists():
            self.img_id2camera=read_pickle(str(cache_fn))
        else:
            for img_id in self.get_img_ids():
                self.get_K(img_id)
            cache_fn.parent.mkdir(parents=True,exist_ok=True)
            save_pickle(self.img_id2camera,str(cache_fn))

        self.model_verts=None
        cache_verts_fn = Path(f'data/shapenet/shapenet_cache/{self.category}-{self.model_name}-{self.render_setting}-verts.pkl')
        if cache_verts_fn.exists():
            self.model_verts, self.object_center, self.object_diameter = read_pickle(str(cache_verts_fn))
        else:
            self.model_verts = self.parse_model_verts()
            min_pt = np.min(self.model_verts, 0)
            max_pt = np.max(self.model_verts, 0)
            self.object_center = (max_pt + min_pt) / 2
            self.object_diameter = np.linalg.norm(max_pt - min_pt)
            save_pickle([self.model_verts, self.object_center, self.object_diameter], str(cache_verts_fn))

    def parse_model_verts(self):
        raise NotImplementedError
        import open3d
        SHAPENET_ROOT='/home/liuyuan/data/ShapeNetCore.v2'
        mesh = open3d.io.read_triangle_mesh(f'{SHAPENET_ROOT}/{self.category}/{self.model_name}/models/model_normalized.obj')
        return np.asarray(mesh.vertices,np.float32)

    def get_image(self, img_id, ref_mode=False):
        try:
            return imread(f'{SHAPENET_RENDER_ROOT}/{self.render_setting}/{self.category}/{self.model_name}/{img_id}.png')[:,:,:3]
        except ValueError:
            print(f'{SHAPENET_RENDER_ROOT}/{self.render_setting}/{self.category}/{self.model_name}/{img_id}.png')
            import ipdb; ipdb.set_trace()

    def get_K(self, img_id):
        if img_id in self.img_id2camera:
            pose, K = self.img_id2camera[img_id]
        else:
            pose, K = read_pickle(f'{SHAPENET_RENDER_ROOT}/{self.render_setting}/{self.category}/{self.model_name}/{img_id}-camera.pkl')
            self.img_id2camera[img_id] = (pose, K)
        return np.copy(K)

    def get_pose(self, img_id):
        if img_id in self.img_id2camera:
            pose, K = self.img_id2camera[img_id]
        else:
            pose, K = read_pickle(f'{SHAPENET_RENDER_ROOT}/{self.render_setting}/{self.category}/{self.model_name}/{img_id}-camera.pkl')
            self.img_id2camera[img_id] = (pose, K)
        return np.copy(pose)

    def get_img_ids(self):
        return [str(k) for k in range(self.render_num)]

    def get_mask(self, img_id):
        mask = imread(f'{SHAPENET_RENDER_ROOT}/{self.render_setting}/{self.category}/{self.model_name}/{img_id}.png')[:,:,3]
        return (mask>0).astype(np.float32)

class NormalizedDatabase(BaseDatabase):
    def get_image(self, img_id):
        return self.database.get_image(img_id)

    def get_K(self, img_id):
        return self.database.get_K(img_id)

    def get_pose(self, img_id):
        pose = self.database.get_pose(img_id)
        return normalize_pose(pose, self.scale, self.offset)

    def get_img_ids(self):
        return self.database.get_img_ids()

    def get_mask(self, img_id):
        return self.database.get_mask(img_id)

    def __init__(self, database: BaseDatabase):
        super().__init__("norm/" + database.database_name)
        self.database = database
        center = get_object_center(self.database)
        diameter = get_diameter(self.database)

        self.scale = 2/diameter
        self.offset = - self.scale * center