import os
import urllib.request
import zipfile

samples_dir = "samples"
os.makedirs(samples_dir, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 1. 经典单张目标检测图片
images = [
    ("01_bus_and_pedestrians.jpg", "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"),
    ("02_sports_persons.jpg", "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"),
    ("03_dog_and_bicycle.jpg", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"),
    ("04_horses_in_meadow.jpg", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg"),
    ("05_person_and_beach.jpg", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg"),
    ("06_eagle_flying.jpg", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg"),
    ("07_people_with_kite.jpg", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/kite.jpg"),
    ("08_giraffe_savannah.jpg", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/giraffe.jpg"),
]

for fname, url in images:
    fpath = os.path.join(samples_dir, fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp, open(fpath, "wb") as out_f:
            out_f.write(resp.read())
        print(f"[SUCCESS] Downloaded {fname} ({os.path.getsize(fpath)} bytes)")
    except Exception as e:
        print(f"[FAILED] Failed to download {fname}: {e}")

# 2. 下载并解压 Ultralytics 官方 COCO8 迷你微缩训练样本
coco8_zip = os.path.join(samples_dir, "coco8.zip")
coco8_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
try:
    req = urllib.request.Request(coco8_url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp, open(coco8_zip, "wb") as out_f:
        out_f.write(resp.read())
    print(f"[SUCCESS] Downloaded coco8.zip ({os.path.getsize(coco8_zip)} bytes)")

    with zipfile.ZipFile(coco8_zip, "r") as z:
        z.extractall(samples_dir)
    print("[SUCCESS] Extracted coco8 mini dataset into samples/coco8")
    if os.path.exists(coco8_zip):
        os.remove(coco8_zip)
except Exception as e:
    print(f"[INFO] coco8 dataset download skipped: {e}")

# 3. 生成 classes.txt
classes_file = os.path.join(samples_dir, "classes.txt")
with open(classes_file, "w", encoding="utf-8") as f:
    f.write("person\nbus\ncar\ndog\nbicycle\nhorse\neagle\nkite\ngiraffe\n")

print(f"\nAll sample assets ready in '{samples_dir}' directory!")
