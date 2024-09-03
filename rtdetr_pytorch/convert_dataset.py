# This Python code converts a dataset in YOLO format into the COCO format. 
# The COCO dataset is saved as a JSON file in the output directory.

import json
import os
import sys
import shutil
from PIL import Image, ImageDraw
from tqdm import tqdm



# For DEBUG
def draw_bbox(image, bbox, output_dir):
  x1, y1, x2, y2 = map(int, bbox)
  print(x1, y1, x2, y2)
  
  # Draw the bounding box
  draw = ImageDraw.Draw(image)
  draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
  
  # save the image
  image.save(output_dir+"/"+image_path.split('/')[-1])


def create_dataset(input_dir, output_dir, type_='train', categories=None):
  # Define the COCO dataset dictionary
  coco_dataset = {
      "info": {},
      "licenses": [],
      "categories": categories,
      "images": [],
      "annotations": []
  }
  image_dir = os.listdir(input_dir+f"/{type_}/images")
  image_dir = sorted(image_dir)
  
  id_ = 0
  for image_file in tqdm(image_dir):
      
      # Load the image and get its dimensions
      image_path = os.path.join(input_dir+f"/{type_}/images", image_file)
      image = Image.open(image_path)
      width, height = image.size

      # copy image file to new dataset
      shutil.copyfile(image_path, output_dir+f"/{type_}/{image_file}")
      
      # Add the image to the COCO dataset
      image_dict = {
          "id": id_,
          "width": width,
          "height": height,
          "file_name": image_file
      }
      coco_dataset["images"].append(image_dict)
      
      # Load the bounding box annotations for the image
      with open(os.path.join(input_dir+f"/{type_}/labels", f'{image_file.split(".")[0]}.txt')) as f:
          annotations = f.readlines()
      
      # Loop through the annotations and add them to the COCO dataset
      for ann in annotations:
          x, y, w, h = map(float, ann.strip().split()[1:])

          x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
          x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)


          # Debug
          #draw_bbox(image.copy(), [x_min,y_min,x_max,y_max], output_dir=output_dir)

          ann_dict = {
              "id": len(coco_dataset["annotations"]),
              "image_id": id_,
              "category_id": int(ann.strip().split()[0]),
              "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
              "area": (x_max - x_min) * (y_max - y_min),
              "iscrowd": 0
          }
          coco_dataset["annotations"].append(ann_dict)

      # increment id
      id_ += 1

  # Save the COCO dataset to a JSON file
  with open(os.path.join(output_dir, f'annotations/{type_}_annotations.json'), 'w') as f:
      json.dump(coco_dataset, f)

if __name__ == '__main__':
  # Set the paths for the input and output directories
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]

  # Define the categories for the COCO dataset
  categories = [{'id': 0, 'name': 'bicycle'}, {'id': 1, 'name': 'bus'}, {'id': 2, 'name': 'car'}, {'id': 3, 'name': 'truck'}, {'id': 4, 'name': 'motorcycle'}, {'id': 5, 'name': 'person'}]

  # Create folder if not exist
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(output_dir+f"/train")
    os.makedirs(output_dir+f"/valid")
    os.makedirs(output_dir+f"/annotations")

  for type_ in ['train', 'valid']:
    print(f"Converting {type_} dataset...")
    create_dataset(input_dir, output_dir, type_=type_, categories=categories)
