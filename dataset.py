import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image

class PartDiscoveryDataset(Dataset):
    """
    Unified Dataset loader for Unsupervised Part Discovery.
    Supports: 'CelebA', 'CUB', and 'PartImageNet'.
    Based on MPAE formats.
    """
    def __init__(self, root, dataset_name, split='train', img_size=224):
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.img_size = img_size
        
        # Standard DINOv2 / ImageNet transform
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.database = []

        if self.dataset_name == 'celeba':
            self._load_celeba()
        elif self.dataset_name == 'cub':
            self._load_cub()
        elif self.dataset_name == 'partimagenet':
            self._load_partimagenet()
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported. Use CelebA, CUB, or PartImageNet.")

    def _load_celeba(self):
        """Loads MAFL/CelebA landmark dataset"""
        anno_file = 'cele_train_lm.txt' if self.split == 'train' else 'MAFL_test_lm.txt'
        anno_path = os.path.join(self.root, anno_file)
        
        if not os.path.exists(anno_path):
            print(f"Warning: CelebA annotation file not found at {anno_path}")
            return

        with open(anno_path, 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            info = line.split(',')
            img_path = os.path.join(self.root, 'img_celeba', info[0])
            
            # Format: 5 landmarks (x, y)
            points = np.array([float(x) for x in info[1:11]]).reshape((5, 2))
            
            self.database.append({
                'img_path': img_path,
                'landmarks': points
            })

    def _load_cub(self):
        """Loads CUB-200-2011 dataset with parts"""
        list_file = 'train_list.txt' if self.split == 'train' else 'test_list.txt'
        list_path = os.path.join(self.root, list_file)
        
        if not os.path.exists(list_path):
            print(f"Warning: CUB list file not found at {list_path}")
            return

        with open(list_path, 'r') as f:
            img_list = f.read().splitlines()

        # Load boxes and landmarks mapping
        box_dict = {}
        with open(os.path.join(self.root, 'bounding_boxes.txt'), 'r') as f:
            for line in f.read().splitlines():
                parts = line.split(' ')
                box_dict[parts[0]] = np.array([float(p) for p in parts[1:5]])
                
        landmark_dict = {}
        with open(os.path.join(self.root, "parts", "part_locs.txt"), 'r') as f:
            for line in f.read().splitlines():
                parts = line.split(' ')
                img_id = parts[0]
                if img_id not in landmark_dict:
                    landmark_dict[img_id] = np.zeros((15, 3), dtype=np.float32)
                # part_id, x, y, visible
                landmark_dict[img_id][int(parts[1])-1] = [float(parts[2]), float(parts[3]), float(parts[4])]

        for item in img_list:
            parts = item.split(' ')
            img_id, rel_path = parts[0], parts[1]
            img_path = os.path.join(self.root, 'images', rel_path)
            
            self.database.append({
                'img_path': img_path,
                'box': box_dict.get(img_id, None),
                'landmarks': landmark_dict.get(img_id, None)
            })

    def _load_partimagenet(self):
        """Loads PartImageNet annotations in COCO format"""
        # PartImageNet splits
        if self.split == 'train':
            anno_path = os.path.join(self.root, 'train_train.json')
            if not os.path.exists(anno_path): # Backup path structure
                 anno_path = os.path.join(self.root, 'annotations', 'train', 'train.json')
        else:
            anno_path = os.path.join(self.root, 'train_test.json')
            if not os.path.exists(anno_path):
                 anno_path = os.path.join(self.root, 'annotations', 'test', 'test.json')

        if not os.path.exists(anno_path):
            print(f"Warning: PartImageNet JSON not found at {anno_path}")
            return

        print(f"Loading PartImageNet COCO format from {anno_path}...")
        self.coco = COCO(anno_path)
        
        for img_id, img_info in self.coco.imgs.items():
            # Resolve image path
            img_name = img_info['file_name'].replace('\\', '/').split('/')[-1]
            
            img_path = os.path.join(self.root, self.split, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(self.root, 'images', self.split, img_name)
                
            self.database.append({
                'img_path': img_path,
                'img_id': img_id,
                'img_info': img_info
            })

    def _get_partimagenet_masks(self, item):
        img_id = item['img_id']
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        num_kps = len(self.coco.cats)
        
        mask_tensor = torch.zeros(size=(num_kps, item['img_info']['height'], item['img_info']['width']))
        
        cat_ids = sorted(list(self.coco.cats.keys()))
        cat_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        
        for ann in anns:
            if ann.get("area", 1) == 0 or ann.get("iscrowd", 0) == 1:
                continue
            cat_idx = cat_to_idx[ann['category_id']]
            mask = torch.as_tensor(self.coco.annToMask(ann), dtype=torch.float32)
            mask_tensor[cat_idx] += mask
            
        mask_tensor = torch.clamp(mask_tensor, 0, 1)
        
        # Resize mask to target size nearest neighbor
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0), 
            size=(self.img_size, self.img_size), 
            mode='nearest'
        ).squeeze(0)
        return mask_resized

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        item = self.database[idx]
        img_path = item['img_path']

        # Load & Convert Image
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
        except:
            # Fallback for broken images during development
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.img_size, self.img_size

        pil_img = Image.fromarray(img)
        img_tensor = self.transform(pil_img)

        meta = {
            'image': img_tensor,
            'image_path': img_path,
            'original_size': (orig_w, orig_h)
        }

        # Add dataset-specific metadata / annotations
        if self.dataset_name == 'celeba':
            marks = item['landmarks'].copy()
            # Scale coordinates
            marks[:, 0] = marks[:, 0] / orig_w * self.img_size
            marks[:, 1] = marks[:, 1] / orig_h * self.img_size
            meta['landmarks'] = torch.from_numpy(marks).float()
            
        elif self.dataset_name == 'cub':
            marks = item['landmarks'].copy()
            marks[:, 0] = marks[:, 0] / orig_w * self.img_size
            marks[:, 1] = marks[:, 1] / orig_h * self.img_size
            meta['landmarks'] = torch.from_numpy(marks).float()
            
            box = item['box'].copy()
            box[:2] = box[:2] / [orig_w, orig_h] * self.img_size
            box[2:] = box[2:] / [orig_w, orig_h] * self.img_size
            meta['box'] = torch.from_numpy(box).float()
            
        elif self.dataset_name == 'partimagenet':
            meta['mask'] = self._get_partimagenet_masks(item)

        return meta

class CelebA_Dataset_MPAE(Dataset):
    def __init__(self, root, is_train, Image_size=448):
        self.Image_size = Image_size
        self.is_train = is_train
        self.root = root

        self.Fraction = 1.0 # default values from Ref_MPAE
        self.Translation_Factor = 0.05
        self.Rotation_Factor = 5.0
        self.Scale_Factor = 0.05
        self.Flip = 0.5

        transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        if not is_train:
            aug_Transform = [
                transforms.Resize((self.Image_size, self.Image_size), antialias=True),
                ]
            self.Transform = transforms.Compose(aug_Transform + transform)
        else:
            aug_Transform = [
                transforms.Resize((self.Image_size, self.Image_size), antialias=True),
                transforms.RandomHorizontalFlip(p=self.Flip),
                transforms.ColorJitter(),
                transforms.RandomAffine(degrees=self.Rotation_Factor, translate=(self.Translation_Factor, self.Translation_Factor),
                                        scale=(1.0 - self.Scale_Factor, 1.0 + self.Scale_Factor)),
                # transforms.RandomCrop(self.Image_size)  # Keep it simple for now, we usually only care about test
                ]
            self.Transform = transforms.Compose(aug_Transform + transform)

        self.annotation_file = os.path.join(root, 'cele_train_lm.txt' if is_train else 'MAFL_test_lm.txt')
        self.landmarks_file = os.path.join(root, 'list_landmarks_align_celeba.txt')

        self.database = self.get_file_information()

    def get_file_information(self):
        # 1. Get split images
        split_images = []
        with open(self.annotation_file) as f:
            lines = f.read().splitlines()
            for line in lines:
                parts = line.replace(',', ' ').split()
                if len(parts) > 0:
                    split_images.append(parts[0])

        # 2. Get Aligned Landmarks
        aligned_points = {}
        if os.path.exists(self.landmarks_file):
            with open(self.landmarks_file) as f:
                lines = f.read().splitlines()
            for line in lines[2:]:  # skip count and header
                parts = line.split()
                if len(parts) >= 11:
                    img_name = parts[0]
                    pts = np.array([float(x) for x in parts[1:11]]).reshape((5, 2))
                    aligned_points[img_name] = pts

        Data_base = []
        for img_name in split_images:
            if img_name in aligned_points:
                temp_path = os.path.join(self.root, 'img_celeba', img_name)
                Data_base.append({'Img': temp_path, 'points': aligned_points[img_name]})

        print(f"Loaded {len(Data_base)} CelebA images for split={ 'train' if self.is_train else 'test' } using {os.path.basename(self.annotation_file)}")
        return Data_base

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        import copy
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        points = db_slic['points']

        Img = cv2.imread(Img_path)
        if Img is None:
            # fallback for broken images
            Img = np.zeros((self.Image_size, self.Image_size, 3), dtype=np.uint8)
        
        Img_shape = Img.shape

        Img = cv2.resize(Img, (self.Image_size, self.Image_size))
        points[:, 0] = points[:, 0] / Img_shape[1] * self.Image_size
        points[:, 1] = points[:, 1] / Img_shape[0] * self.Image_size

        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        input = Image.fromarray(Img)
        if self.Transform is not None:
            input_tensor = self.Transform(input)

        # Make it compatible with our `PartDiscoveryDataset` expected output
        meta = {
            'image': input_tensor,
            'image_path': Img_path,
            'original_size': (Img_shape[1], Img_shape[0]),
            'landmarks': torch.from_numpy(points).float()  # Add visible flag in evaluate
        }

        # Need to append `visible` flag to landmarks (5, 3) to match our evaluation
        # In CelebA all 5 points are visible by default
        marks = meta['landmarks']
        marks_with_vis = torch.cat([marks, torch.ones((5, 1))], dim=1)
        meta['landmarks'] = marks_with_vis

        return meta


class PartImageNet_OOD_Dataset(Dataset):
    """
    Dataloader for PartImageNet-OOD (Out-of-Distribution split).

    Dataset layout:
        <root>/train.json, val.json, test.json   — COCO-format annotations
        <root>/train/train/<synset>/<file>.JPEG  — training images
        <root>/val/val/<synset>/<file>.JPEG      — validation images
        <root>/test/test/<synset>/<file>.JPEG    — test images

    Annotation structure:
        40 fine-grained part categories (e.g. 'Bird Head', 'Car Tier')
        grouped into 11 supercategories:
        Quadruped, Biped, Fish, Bird, Snake, Reptile,
        Car, Bicycle, Boat, Aeroplane, Bottle

    Returns (all splits):
        image         — normalized tensor (3, img_size, img_size)
        image_path    — absolute path string
        original_size — (orig_w, orig_h)
        supercategory — string, e.g. 'Bird'
        supercat_idx  — int index into self.super_categories list

    Additionally for val/test splits:
        mask          — float32 tensor (num_parts, img_size, img_size)
                        per-part binary masks resized with nearest-neighbor.
                        Channel i corresponds to fine category id i.
    """

    # Canonical ordering of the 11 supercategories
    SUPER_CATEGORIES = [
        'Aeroplane', 'Bicycle', 'Biped', 'Bird', 'Boat',
        'Bottle', 'Car', 'Fish', 'Quadruped', 'Reptile', 'Snake',
    ]

    def __init__(self, root, split='test', img_size=448):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size

        # Standard DINOv2 / ImageNet normalization (matches rest of Part_Discovery)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        anno_file = os.path.join(root, f'{split}.json')
        if not os.path.exists(anno_file):
            print(f"Warning: PartImageNet_OOD annotation file not found at {anno_file}")
            self.database = []
            self.num_parts = 0
            self.super_categories = self.SUPER_CATEGORIES
            self.supercat_to_idx = {s: i for i, s in enumerate(self.super_categories)}
            return

        self._load(anno_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_annotations(self):
        """Fix degenerate 4-point bbox annotations into valid polygons (from Ref_MPAE)."""
        import copy
        json_dict = copy.deepcopy(self.coco.dataset)
        for ann in json_dict['annotations']:
            if ann.get('area', 1) == 0 or ann.get('iscrowd', 0) == 1:
                continue
            for poly_num, seg in enumerate(ann['segmentation']):
                if len(seg) == 4:
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = x1 + w, y1 + h
                    ann['segmentation'][poly_num] = [x1, y1, x1, y2, x2, y2, x2, y1]
        self.coco.dataset = copy.deepcopy(json_dict)
        self.coco.createIndex()

    def _load(self, anno_file):
        self.coco = COCO(anno_file)
        self._preprocess_annotations()

        # ---- Category metadata ----
        # num_parts = total number of fine-grained part categories
        self.num_parts = len(self.coco.cats)

        # Build sorted supercategory list (deterministic ordering)
        self.super_categories = sorted(
            list(dict.fromkeys(
                c['supercategory'] for c in self.coco.cats.values()
            ))
        )
        self.supercat_to_idx = {s: i for i, s in enumerate(self.super_categories)}

        # Map fine category id -> supercategory string
        self.cat_id_to_supercat = {
            cat_id: cat['supercategory']
            for cat_id, cat in self.coco.cats.items()
        }

        # ---- Build per-image database ----
        self.database = []
        for img_id, img_info in self.coco.imgs.items():
            fn = img_info['file_name'].replace('\\', '/').split('/')[-1]
            synset = fn.split('_')[0]

            # Images live at <root>/<split>/<split>/<synset>/<file>
            img_path = os.path.join(self.root, self.split, self.split, synset, fn)

            # Get supercategory from annotations (use first valid annotation)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid_cats = [
                ann['category_id'] for ann in anns
                if ann.get('area', 1) > 0 and not ann.get('iscrowd', 0)
            ]
            if not valid_cats:
                continue  # skip images with no valid annotations

            supercat = self.cat_id_to_supercat[valid_cats[0]]

            self.database.append({
                'img_path': img_path,
                'img_id': img_id,
                'img_info': img_info,
                'supercategory': supercat,
                'fine_category_ids': list(dict.fromkeys(valid_cats)),
            })

    def _get_mask(self, img_id, orig_h, orig_w):
        """
        Build a (num_parts, img_size, img_size) float32 mask tensor
        with one binary channel per fine-grained part category.
        """
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask_tensor = torch.zeros(self.num_parts, orig_h, orig_w, dtype=torch.float32)

        for ann in anns:
            if ann.get('area', 1) == 0 or ann.get('iscrowd', 0) == 1:
                continue
            cat_id = ann['category_id']
            part_mask = torch.as_tensor(
                self.coco.annToMask(ann), dtype=torch.float32
            )
            mask_tensor[cat_id] = torch.clamp(mask_tensor[cat_id] + part_mask, 0, 1)

        # Resize to img_size using nearest-neighbor (preserves binary values)
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),       # (1, num_parts, H, W)
            size=(self.img_size, self.img_size),
            mode='nearest',
        ).squeeze(0)                        # (num_parts, img_size, img_size)

        return mask_resized

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        item = self.database[idx]
        img_path = item['img_path']
        img_id = item['img_id']
        img_info = item['img_info']
        supercat = item['supercategory']

        # Load image
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
        except Exception:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.img_size, self.img_size

        pil_img = Image.fromarray(img)
        img_tensor = self.transform(pil_img)

        meta = {
            'image': img_tensor,
            'image_path': img_path,
            'original_size': (orig_w, orig_h),
            'supercategory': supercat,
            'supercat_idx': self.supercat_to_idx[supercat],
            'fine_category_ids': item['fine_category_ids'],
        }

        # Provide segmentation masks for val / test splits
        if self.split in ('val', 'test'):
            meta['mask'] = self._get_mask(img_id, img_info['height'], img_info['width'])

        return meta


class PartImageNet_Seg_Dataset(Dataset):
    """
    Dataloader for PartImageNet-Seg (Segmentation split).

    Dataset layout:
        <root>/annotations/{split}_whole/{split}.json  — COCO-format annotations
        <root>/images/{split}/<file>.JPEG               — images (flat, no synset subdir)

    Annotation structure:
        158 ImageNet class categories (e.g. 'n01440764', 'n02422699')
        grouped into 11 supercategories:
        Quadruped, Biped, Fish, Bird, Snake, Reptile,
        Car, Bicycle, Boat, Aeroplane, Bottle

    Returns (all splits):
        image         — normalized tensor (3, img_size, img_size)
        image_path    — absolute path string
        original_size — (orig_w, orig_h)
        supercategory — string, e.g. 'Bird'
        supercat_idx  — int index into self.super_categories list

    Additionally for val/test splits:
        mask          — float32 tensor (num_parts, img_size, img_size)
                        per-part binary masks resized with nearest-neighbor.
                        Channel i corresponds to category id i.
    """

    # Same 11 supercategories as OOD
    SUPER_CATEGORIES = [
        'Aeroplane', 'Bicycle', 'Biped', 'Bird', 'Boat',
        'Bottle', 'Car', 'Fish', 'Quadruped', 'Reptile', 'Snake',
    ]

    def __init__(self, root, split='test', img_size=448):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = img_size

        # Standard DINOv2 / ImageNet normalization (matches rest of Part_Discovery)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Seg annotations live at <root>/annotations/<split>_whole/<split>.json
        anno_file = os.path.join(root, 'annotations', f'{split}_whole', f'{split}.json')
        if not os.path.exists(anno_file):
            print(f"Warning: PartImageNet_Seg annotation file not found at {anno_file}")
            self.database = []
            self.num_parts = 0
            self.super_categories = self.SUPER_CATEGORIES
            self.supercat_to_idx = {s: i for i, s in enumerate(self.super_categories)}
            return

        self._load(anno_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_annotations(self):
        """Fix degenerate 4-point bbox annotations into valid polygons (from Ref_MPAE)."""
        import copy
        json_dict = copy.deepcopy(self.coco.dataset)
        for ann in json_dict['annotations']:
            if ann.get('area', 1) == 0 or ann.get('iscrowd', 0) == 1:
                continue
            for poly_num, seg in enumerate(ann['segmentation']):
                if len(seg) == 4:
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = x1 + w, y1 + h
                    ann['segmentation'][poly_num] = [x1, y1, x1, y2, x2, y2, x2, y1]
        self.coco.dataset = copy.deepcopy(json_dict)
        self.coco.createIndex()

    def _load(self, anno_file):
        self.coco = COCO(anno_file)
        self._preprocess_annotations()

        # ---- Category metadata ----
        self.num_parts = len(self.coco.cats)

        # Build sorted supercategory list (deterministic ordering)
        self.super_categories = sorted(
            list(dict.fromkeys(
                c['supercategory'] for c in self.coco.cats.values()
            ))
        )
        self.supercat_to_idx = {s: i for i, s in enumerate(self.super_categories)}

        # Map category id -> supercategory string
        self.cat_id_to_supercat = {
            cat_id: cat['supercategory']
            for cat_id, cat in self.coco.cats.items()
        }

        # ---- Build per-image database ----
        self.database = []
        for img_id, img_info in self.coco.imgs.items():
            fn = img_info['file_name'].replace('\\', '/').split('/')[-1]

            # Seg images live at <root>/images/<split>/<file>.JPEG (flat layout)
            img_path = os.path.join(self.root, 'images', self.split, fn)

            # Get supercategory from annotations (use first valid annotation)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid_cats = [
                ann['category_id'] for ann in anns
                if ann.get('area', 1) > 0 and not ann.get('iscrowd', 0)
            ]
            if not valid_cats:
                continue  # skip images with no valid annotations

            supercat = self.cat_id_to_supercat[valid_cats[0]]

            self.database.append({
                'img_path': img_path,
                'img_id': img_id,
                'img_info': img_info,
                'supercategory': supercat,
                'fine_category_ids': list(dict.fromkeys(valid_cats)),
            })

    def _get_mask(self, img_id, orig_h, orig_w):
        """
        Build a (num_parts, img_size, img_size) float32 mask tensor
        with one binary channel per category.
        """
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask_tensor = torch.zeros(self.num_parts, orig_h, orig_w, dtype=torch.float32)

        for ann in anns:
            if ann.get('area', 1) == 0 or ann.get('iscrowd', 0) == 1:
                continue
            cat_id = ann['category_id']
            part_mask = torch.as_tensor(
                self.coco.annToMask(ann), dtype=torch.float32
            )
            mask_tensor[cat_id] = torch.clamp(mask_tensor[cat_id] + part_mask, 0, 1)

        # Resize to img_size using nearest-neighbor (preserves binary values)
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),       # (1, num_parts, H, W)
            size=(self.img_size, self.img_size),
            mode='nearest',
        ).squeeze(0)                        # (num_parts, img_size, img_size)

        return mask_resized

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        item = self.database[idx]
        img_path = item['img_path']
        img_id = item['img_id']
        img_info = item['img_info']
        supercat = item['supercategory']

        # Load image
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
        except Exception:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.img_size, self.img_size

        pil_img = Image.fromarray(img)
        img_tensor = self.transform(pil_img)

        meta = {
            'image': img_tensor,
            'image_path': img_path,
            'original_size': (orig_w, orig_h),
            'supercategory': supercat,
            'supercat_idx': self.supercat_to_idx[supercat],
            'fine_category_ids': item['fine_category_ids'],
        }

        # Provide segmentation masks for val / test splits
        if self.split in ('val', 'test'):
            meta['mask'] = self._get_mask(img_id, img_info['height'], img_info['width'])

        return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test dataset loaders")
    parser.add_argument('--data_dir', type=str, default='./data', help='Base directory for datasets')
    args = parser.parse_args()

    print("Testing dataset initializations...")
    for ds_name in ['CelebA', 'CUB']:
        path = os.path.join(args.data_dir, ds_name)
        if os.path.exists(path):
            ds = PartDiscoveryDataset(path, ds_name)
            print(f"Loaded {ds_name}: {len(ds)} samples")
        else:
            print(f"Path not found for {ds_name}: {path}")

    ood_path = os.path.join(args.data_dir, 'PartImageNet_OOD')
    if os.path.exists(ood_path):
        ds_ood = PartImageNet_OOD_Dataset(ood_path, split='test')
        print(f"Loaded PartImageNet_OOD: {len(ds_ood)} images, {ds_ood.num_parts} parts")

    seg_path = os.path.join(args.data_dir, 'PartImageNet_Seg', 'PartImageNet')
    if os.path.exists(seg_path):
        ds_seg = PartImageNet_Seg_Dataset(seg_path, split='test')
        print(f"Loaded PartImageNet_Seg: {len(ds_seg)} images, {ds_seg.num_parts} categories")
    print("Done!")

