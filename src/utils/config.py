"""
全局配置管理器 (Configuration Manager)
"""
import json
import os
from typing import Any, Dict


class ConfigManager:
    DEFAULT_CONFIG = {
        "recent_projects": [],
        "last_project_path": "",
        "theme": "dark",
        "default_model": "yolov8n.pt",
        "default_epochs": 100,
        "default_batch": 16,
        "default_imgsz": 640,
        "auto_save_interval": 10,
        "show_crosshair": True,
        "default_confidence_threshold": 0.25,
        "default_iou_threshold": 0.45,
        "custom_classes": ["target"]
    }

    def __init__(self, config_path: str = None):
        if config_path is None:
            home_dir = os.path.expanduser("~")
            app_dir = os.path.join(home_dir, ".yolo_studio")
            os.makedirs(app_dir, exist_ok=True)
            self.config_path = os.path.join(app_dir, "config.json")
        else:
            self.config_path = config_path

        self.config: Dict[str, Any] = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 合并默认值
                    merged = self.DEFAULT_CONFIG.copy()
                    merged.update(data)
                    return merged
            except Exception:
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()

    def save_config(self) -> bool:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value
        self.save_config()

    def add_recent_project(self, project_path: str) -> None:
        recent = self.config.get("recent_projects", [])
        if project_path in recent:
            recent.remove(project_path)
        recent.insert(0, project_path)
        self.config["recent_projects"] = recent[:10]  # 保留最近10个
        self.config["last_project_path"] = project_path
        self.save_config()
