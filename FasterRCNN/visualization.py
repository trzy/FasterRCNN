from .dataset import VOC
from .models import region_proposal_network
from .models import intersection_over_union

import imageio
from PIL import Image, ImageDraw

def draw_rectangle(image, x_min, y_min, x_max, y_max, color, thickness = 4):
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], outline = color, width = thickness)

def draw_filled_rectangle(image, x_min, y_min, x_max, y_max, color):
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], fill = color)

def show_annotated_image(voc, filename, draw_anchor_intersections = False, image_input_map = None, anchor_map = None):
  # Load image
  filepath = voc.get_full_path(filename = filename)
  info = voc.get_image_description(path = filepath)
  data = imageio.imread(filepath, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  
  # Because we loaded it ourselves, we are responsible for rescaling it
  image = image.resize((info.width, info.height), resample = Image.BILINEAR)
  
  # Draw ground truth boxes
  _draw_ground_truth_boxes(image = image, boxes = info.get_boxes())
  
  # Draw anchor boxes and intersection areas, if requested
  if draw_anchor_intersections:
    _draw_anchor_box_intersections(image = image, image_input_map = image_input_map, anchor_map = anchor_map, ground_truth_boxes = info.get_boxes())
  
  # Display image
  image.show()

def _draw_ground_truth_boxes(image, boxes):
  # Draw green boxes
  ctx = ImageDraw.Draw(image)
  for box in boxes:
    draw_rectangle(image, x_min = box.x_min, y_min = box.y_min, x_max = box.x_max, y_max = box.y_max, color = (0, 255, 0, 255))

def _draw_anchor_box_intersections(image, image_input_map, anchor_map, ground_truth_boxes):
  anchor_boxes = region_proposal_network.compute_all_anchor_boxes(image_input_map = image_input_map, anchor_map = anchor_map)
  for y in range(anchor_boxes.shape[0]):
    for x in range(anchor_boxes.shape[1]):
      anchors_per_location = int(anchor_boxes.shape[2] / 4)
      assert int(anchor_boxes.shape[2]) % int(4) == 0
      for anchor_num in range(anchors_per_location):
        # Extract box
        box = anchor_boxes[y, x, anchor_num * 4 : anchor_num * 4 + 4]
        center_y = box[0]
        center_x = box[1]
        height = box[2]
        width = box[3]
        anchor_y_min = center_y - 0.5 * height
        anchor_x_min = center_x - 0.5 * width
        anchor_y_max = center_y + 0.5 * height
        anchor_x_max = center_x + 0.5 * width

        # Does it intersect with any ground truth box?
        for box in ground_truth_boxes:
          box1 = (anchor_y_min, anchor_x_min, anchor_y_max, anchor_x_max)
          box2 = (box.y_min, box.x_min, box.y_max, box.x_max)

          iou = intersection_over_union.intersection_over_union(box1 = box1, box2 = box2)
          if iou > 0:
            # Render intersection area
            intersection_box = intersection_over_union.intersection(box1 = box1, box2 = box2)
            draw_filled_rectangle(image = image, y_min = intersection_box[0], x_min = intersection_box[1], y_max = intersection_box[2], x_max = intersection_box[3], color = (255, 0, 0, 64))
            
            # Draw the anchor box outline
            draw_rectangle(image, x_min = anchor_x_min, y_min = anchor_y_min, x_max = anchor_x_max, y_max = anchor_y_max, color = (255, 0, 0, 255))

