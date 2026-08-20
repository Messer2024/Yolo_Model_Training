"""
数据集管理器 (Dataset Manager)
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import shutil
import random

try:
    import yaml
except ImportError:
    # 简易纯 Python YAML 解析与输出 Fallback
    class SimpleYaml:
        @staticmethod
        def safe_load(stream):
            data = {}
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if v.isdigit():
                        data[k] = int(v)
                    else:
                        data[k] = v
            return data

        @staticmethod
        def dump(data, stream, **kwargs):
            for k, v in data.items():
                if isinstance(v, dict):
                    stream.write(f"{k}:\n")
                    for sub_k, sub_v in v.items():
                        stream.write(f"  {sub_k}: {sub_v}\n")
                elif isinstance(v, list):
                    stream.write(f"{k}:\n")
                    for item in v:
                        stream.write(f"  - {item}\n")
                else:
                    stream.write(f"{k}: {v}\n")

    yaml = SimpleYaml()

from src.core.annotation import BoundingBox, get_class_color
from src.utils.logger import logger

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetManager:
    """负责数据集的项目加载、标注持久化、数据划分、健康体检与 data.yaml 生成"""

    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir: str = ""
        self.image_files: List[str] = []
        self.labels_map: Dict[str, List[BoundingBox]] = {}  # {img_path: [BoundingBox, ...]}
        self.class_names: List[str] = ["target"]
        self.class_colors: Dict[int, str] = {0: get_class_color(0)}
        self.current_image_index: int = 0

        if project_dir and os.path.exists(project_dir):
            self.load_project(project_dir)

    def load_project(self, project_dir: str) -> bool:
        """加载项目目录中的图片与标注文件"""
        if not os.path.exists(project_dir):
            return False

        self.project_dir = os.path.abspath(project_dir)
        self.image_files = []
        self.labels_map = {}

        # 1. 扫描 classes.txt 或 data.yaml
        classes_file = os.path.join(self.project_dir, "classes.txt")
        data_yaml_file = os.path.join(self.project_dir, "data.yaml")

        if os.path.exists(classes_file):
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        self.class_names = lines
            except Exception as e:
                logger.warning(f"读取 classes.txt 失败: {e}")
        elif os.path.exists(data_yaml_file):
            try:
                with open(data_yaml_file, "r", encoding="utf-8") as f:
                    data_cfg = yaml.safe_load(f)
                    names = data_cfg.get("names", [])
                    if isinstance(names, dict):
                        self.class_names = [names[k] for k in sorted(names.keys())]
                    elif isinstance(names, list) and names:
                        self.class_names = names
            except Exception as e:
                logger.warning(f"读取 data.yaml 失败: {e}")

        # 更新颜色映射
        self.class_colors = {i: get_class_color(i) for i in range(len(self.class_names))}

        # 2. 扫描所有图片
        for root, _, files in os.walk(self.project_dir):
            if "runs" in root or ".venv" in root or "venv" in root or "__pycache__" in root:
                continue
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    img_path = os.path.join(root, f)
                    self.image_files.append(img_path)

        # 3. 加载对应的标注文件 (.txt)
        for img_path in self.image_files:
            txt_path = self._get_label_path(img_path)
            boxes = []
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        for line in f:
                            box = BoundingBox.from_yolo_str(line)
                            if box:
                                boxes.append(box)
                                if box.class_id >= len(self.class_names):
                                    for new_id in range(len(self.class_names), box.class_id + 1):
                                        self.class_names.append(f"class_{new_id}")
                                        self.class_colors[new_id] = get_class_color(new_id)
                except Exception as e:
                    logger.warning(f"读取标注文件 {txt_path} 失败: {e}")
            self.labels_map[img_path] = boxes

        self.current_image_index = 0
        logger.info(f"成功加载项目: {self.project_dir}，共 {len(self.image_files)} 张图像，{len(self.class_names)} 个类别")
        return True

    def _get_label_path(self, img_path: str) -> str:
        """获取图像对应的 .txt 标注文件路径"""
        base, _ = os.path.splitext(img_path)
        direct_txt = base + ".txt"

        if os.path.exists(direct_txt):
            return direct_txt

        if "images" in img_path:
            alt_txt = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
            alt_txt = os.path.splitext(alt_txt)[0] + ".txt"
            if os.path.exists(alt_txt):
                return alt_txt

        return direct_txt

    def save_annotation(self, img_path: str, boxes: List[BoundingBox]) -> bool:
        """将某张图片的标注信息保存为 YOLO 格式 .txt 文件"""
        self.labels_map[img_path] = boxes
        txt_path = self._get_label_path(img_path)

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(b.to_yolo_str() + "\n")
            return True
        except Exception as e:
            logger.error(f"保存标注失败 {txt_path}: {e}")
            return False

    def save_classes(self) -> bool:
        """保存 classes.txt 到项目根目录"""
        if not self.project_dir:
            return False
        classes_file = os.path.join(self.project_dir, "classes.txt")
        try:
            with open(classes_file, "w", encoding="utf-8") as f:
                for name in self.class_names:
                    f.write(f"{name}\n")
            return True
        except Exception as e:
            logger.error(f"保存 classes.txt 失败: {e}")
            return False

    def add_class(self, class_name: str) -> int:
        """添加新类别"""
        class_name = class_name.strip()
        if not class_name:
            return -1
        if class_name in self.class_names:
            return self.class_names.index(class_name)

        new_id = len(self.class_names)
        self.class_names.append(class_name)
        self.class_colors[new_id] = get_class_color(new_id)
        self.save_classes()
        return new_id

    def update_class_name(self, class_id: int, new_name: str) -> bool:
        """重命名类别"""
        if 0 <= class_id < len(self.class_names):
            self.class_names[class_id] = new_name.strip()
            self.save_classes()
            return True
        return False

    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """按比例划分数据集为 train / val / test 并生成标准目录结构与 data.yaml"""
        if not self.image_files:
            return {"success": False, "message": "当前没有图像可供划分"}

        target_dir = output_dir or os.path.join(self.project_dir, "dataset_split")
        os.makedirs(target_dir, exist_ok=True)

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(target_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(target_dir, "labels", split), exist_ok=True)

        shuffled = self.image_files.copy()
        random.seed(42)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        if train_ratio + val_ratio + test_ratio < 0.999:
            n_val = n_total - n_train

        train_imgs = shuffled[:n_train]
        val_imgs = shuffled[n_train:n_train + n_val]
        test_imgs = shuffled[n_train + n_val:]

        split_dict = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }

        for split_name, img_list in split_dict.items():
            for src_img in img_list:
                img_name = os.path.basename(src_img)
                dst_img = os.path.join(target_dir, "images", split_name, img_name)
                if os.path.abspath(src_img) != os.path.abspath(dst_img):
                    try:
                        shutil.copy2(src_img, dst_img)
                    except Exception as e:
                        logger.warning(f"复制图像失败: {e}")

                src_txt = self._get_label_path(src_img)
                dst_txt = os.path.join(target_dir, "labels", split_name, os.path.splitext(img_name)[0] + ".txt")

                if os.path.exists(src_txt):
                    if os.path.abspath(src_txt) != os.path.abspath(dst_txt):
                        try:
                            shutil.copy2(src_txt, dst_txt)
                        except Exception as e:
                            logger.warning(f"复制标注失败: {e}")
                else:
                    boxes = self.labels_map.get(src_img, [])
                    try:
                        with open(dst_txt, "w", encoding="utf-8") as f:
                            for b in boxes:
                                f.write(b.to_yolo_str() + "\n")
                    except Exception as e:
                        logger.warning(f"写入标注失败: {e}")

        yaml_path = os.path.join(target_dir, "data.yaml")
        yaml_content = {
            "path": os.path.abspath(target_dir).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test" if test_imgs else "",
            "nc": len(self.class_names),
            "names": {i: name for i, name in enumerate(self.class_names)}
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        return {
            "success": True,
            "yaml_path": yaml_path,
            "output_dir": target_dir,
            "train_count": len(train_imgs),
            "val_count": len(val_imgs),
            "test_count": len(test_imgs),
            "message": f"数据集划分完成：Train {len(train_imgs)}, Val {len(val_imgs)}, Test {len(test_imgs)}"
        }

    def audit_dataset(self) -> Dict[str, Any]:
        """对当前数据集执行全面质量体检"""
        from agents.dataset_audit_agent import DatasetAuditAgent
        agent = DatasetAuditAgent()

        context = {
            "image_files": self.image_files,
            "labels_map": {
                img: [b.to_dict() for b in boxes]
                for img, boxes in self.labels_map.items()
            },
            "class_names": self.class_names
        }
        return agent.run(context)
