from collections import defaultdict
import os
from pathlib import Path
import xml.etree.ElementTree as ET

class VOC:
  def __init__(self, dataset_dir):
    self.index_to_class_name = self._get_index_to_class_name(dataset_dir)
    train_image_paths = self._get_image_paths(dataset_dir, dataset = "train")
    val_image_paths = self._get_image_paths(dataset_dir, dataset = "val")
    descriptions_per_train_image = { image_path: self._get_image_description(dataset_dir, image_path = image_path) for image_path in train_image_paths }
    descriptions_per_val_image = { image_path: self._get_image_description(dataset_dir, image_path = image_path) for image_path in train_image_paths }
    print(descriptions_per_val_image)

  class Box:
    def __init__(self, xmin, ymin, xmax, ymax):
      self.xmin = xmin
      self.xmax = xmax
      self.ymin = ymin
      self.ymax = ymax

    def __repr__(self):
      return "[x=%d, y=%d, width=%d, height=%d]" % (self.xmin, self.ymin, self.xmax - self.xmin + 1, self.ymax - self.ymin + 1)

    def __str__(self):
      return repr(self)

  class ImageDescription:
    def __init__(self,name, width, height, boxes_by_class_name):
      self.name = name
      self.width = width
      self.height = height
      self.boxes_by_class_name = boxes_by_class_name

    def __repr__(self):
      return "[name=%s, (%d, %d), boxes=%s]" % (self.name, self.width, self.height, self.boxes_by_class_name)

  @staticmethod
  def _get_index_to_class_name(dataset_dir):
    imageset_dir = os.path.join(dataset_dir, "ImageSets", "Main")
    train_classes = set([ os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_train.txt") ])
    val_classes = set([ os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_val.txt") ])
    assert train_classes == val_classes, "Number of training and validation image sets in ImageSets/Main differs. Does your dataset have missing or extraneous files?"
    assert len(train_classes) > 0, "No classes found in ImageSets/Main"
    index_to_class_name = { v[0]: v[1] for v in enumerate(train_classes) }
    return index_to_class_name

  @staticmethod
  def _get_image_paths(dataset_dir, dataset):
    image_list_file = os.path.join(dataset_dir, "ImageSets", "Main", dataset + ".txt")
    with open(image_list_file) as fp:
      basenames = [ line.strip() for line in fp.readlines() ] # strip newlines
    image_paths = [ os.path.join(dataset_dir, "JPEGImages", basename) + ".jpg" for basename in basenames ]
    return image_paths

  @staticmethod
  def _get_image_description(dataset_dir, image_path):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    annotation_file = os.path.join(dataset_dir, "Annotations", basename) + ".xml"
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    assert tree != None, "Failed to parse %s" % annotation_file
    assert len(root.findall("size")) == 1
    size = root.find("size")
    assert len(size.findall("width")) == 1
    assert len(size.findall("height")) == 1
    assert len(size.findall("depth")) == 1
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    depth = int(size.find("depth").text)
    assert depth == 3
    boxes_by_class_name = defaultdict(list)
    for obj in root.findall("object"):
      #TODO: use "difficult" attribute to optionally exclude difficult images?
      assert len(obj.findall("name")) == 1
      assert len(obj.findall("bndbox")) == 1
      class_name = obj.find("name").text
      bndbox = obj.find("bndbox")
      assert len(bndbox.findall("xmin")) == 1
      assert len(bndbox.findall("ymin")) == 1
      assert len(bndbox.findall("xmax")) == 1
      assert len(bndbox.findall("ymax")) == 1
      xmin = int(bndbox.find("xmin").text)
      ymin = int(bndbox.find("ymin").text)
      xmax = int(bndbox.find("xmax").text)
      ymax = int(bndbox.find("ymax").text)
      box = VOC.Box(xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)
      boxes_by_class_name[class_name].append(box)
    return VOC.ImageDescription(name = basename, width = width, height = height, boxes_by_class_name = boxes_by_class_name)
