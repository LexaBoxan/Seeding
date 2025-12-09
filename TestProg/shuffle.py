import os
import random

# Путь к train/images
IMG_DIR = r"E:\_JOB_\_Python\Seeding\dataset\datasetSegV6\train\images"
LBL_DIR = r"E:\_JOB_\_Python\Seeding\dataset\datasetSegV6\train\labels"

# Куда записывать txt-файлы
OUT_DIR = r"E:\_JOB_\_Python\Seeding\dataset\datasetSegV6"

# Сбор изображений
imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Проверяем, что для каждой картинки есть метка
valid_imgs = []
for img in imgs:
    name = os.path.splitext(img)[0] + ".txt"
    lbl_path = os.path.join(LBL_DIR, name)
    if os.path.exists(lbl_path):
        valid_imgs.append(img)
    else:
        print("⚠️ Нет метки для:", img)

# Перемешиваем
random.seed(42)  # повторяемость
valid_imgs = random.sample(valid_imgs, len(valid_imgs))  # равномерное перемешивание


n = len(valid_imgs)
# --- Новые пропорции ---
train_split = int(n * 0.70)
val_split = int(n * 0.90 )

train = valid_imgs[:train_split]
val = valid_imgs[train_split:val_split]
test = valid_imgs[val_split:]

def write_list(name, items):
    file_path = os.path.join(OUT_DIR, name)
    with open(file_path, "w", encoding="utf-8") as f:
        for img in items:
            abs_path = os.path.abspath(os.path.join(IMG_DIR, img))
            f.write(abs_path + "\n")
    print("Created:", file_path)

write_list("train.txt", train)
write_list("val.txt", val)
write_list("test.txt", test)

print(f"✔ Всего: {n} изображений\n"
      f"✔ Train: {len(train)}\n"
      f"✔ Val:   {len(val)}\n"
      f"✔ Test:  {len(test)}")
