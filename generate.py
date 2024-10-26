from pycocotools.coco import COCO
from augment import generate_image
from tqdm import tqdm
import random
import shutil
import os

annotations_file = "D:/COCO/instances_train2017.json"
images_dir = "D:/COCO/train2017"
output_dir = "./dataset"
output_file = "instances.csv"
num_of_instances = 25000
bicycle_meshes_path = "./bikes"

instances = []
os.makedirs(output_dir, exist_ok=True)
coco = COCO(annotations_file)

bicycle_category_id = coco.getCatIds(catNms=['bicycle'])
bicycle_img_ids = coco.getImgIds(catIds=bicycle_category_id)
bicycle_count = len(bicycle_img_ids)

all_img_ids = coco.getImgIds()
non_bicycle_img_ids = list(set(all_img_ids) - set(bicycle_img_ids))
non_bicycle_count = len(non_bicycle_img_ids)

road_img_ids = set()
for category in ['car', 'bus', 'truck', 'traffic light', 'stop sign']:
    new_road_related_ids = coco.getCatIds(catNms=[category])
    new_road_img_ids = coco.getImgIds(catIds=new_road_related_ids)
    for id in new_road_img_ids:
        if not id in bicycle_img_ids:
            road_img_ids.add(id)

road_img_ids = list(road_img_ids)           

if non_bicycle_count >= num_of_instances // 2:
    print(f"Found {non_bicycle_count} non bicycle images, using {num_of_instances // 2}")

    selected_non_bicycle_img_ids = random.sample(non_bicycle_img_ids, num_of_instances // 2)
    for img_id in tqdm(selected_non_bicycle_img_ids, desc="Copying non bicycle imgs"):
        non_bike_img = coco.loadImgs([img_id])[0]
        source_file_path = os.path.join(images_dir, non_bike_img["file_name"])
        target_file_path = os.path.join(output_dir, non_bike_img["file_name"])
        instances.append([non_bike_img["file_name"], 0])
        shutil.copy(source_file_path, target_file_path)

else:
    print(f"Found only {non_bicycle_count} non bicycle images, not enough")

if bicycle_count < num_of_instances // 2:
    print(f"Found {bicycle_count} bicycle images, generating {num_of_instances // 2 - bicycle_count} more")

    for img_id in tqdm(bicycle_img_ids, desc="Copying bicycle imgs"):
        bike_img = coco.loadImgs([img_id])[0]
        source_file_path = os.path.join(images_dir, bike_img["file_name"])
        target_file_path = os.path.join(output_dir, bike_img["file_name"])
        instances.append([bike_img["file_name"], 1])
        shutil.copy(source_file_path, target_file_path)

    for i in tqdm(range(num_of_instances // 2 - bicycle_count), desc="Generating bicycle imgs"):
        road_img = coco.loadImgs([random.choice(road_img_ids)])[0]
        file_name = f"generated_bike_{i}.jpg"
        file_path = os.path.join(output_dir, file_name)
        generate_image(bicycle_meshes_path, os.path.join(images_dir, road_img["file_name"]), file_path)
        instances.append([file_name, 1])

else:
    print(f"Found {bicycle_count} bicycle images, using {num_of_instances // 2}")

    for img_id in tqdm(bicycle_img_ids, desc="Copying bicycle imgs"):
        bike_img = coco.loadImgs([img_id])[0]
        source_file_path = os.path.join(images_dir, bike_img["file_name"])
        target_file_path = os.path.join(output_dir, bike_img["file_name"])
        instances.append([bike_img["file_name"], 1])
        shutil.copy(source_file_path, target_file_path)

print(f"Exporting instances in {output_file}")
random.shuffle(instances)
with open(output_file, mode="w") as file:
    file.write("file_name,has_bike")
    for instance in tqdm(instances):
        file.write(f"\n{instance[0]},{instance[1]}")