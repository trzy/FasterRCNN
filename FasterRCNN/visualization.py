from .dataset import VOC
from .models import region_proposal_network
from .models import intersection_over_union

import imageio
from math import exp
from PIL import Image, ImageDraw

def draw_rectangle(ctx, x_min, y_min, x_max, y_max, color, thickness = 4):
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], outline = color, width = thickness)

def draw_filled_rectangle(ctx, x_min, y_min, x_max, y_max, color):
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], fill = color)

def show_annotated_image(voc, filename, draw_anchor_points = True, draw_anchor_intersections = False, image_input_map = None, anchor_map = None):
  # Load image
  filepath = voc.get_full_path(filename = filename)
  info = voc.get_image_description(path = filepath)
  data = imageio.imread(filepath, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  boxes = info.get_boxes()

  # Because we loaded it ourselves, we are responsible for rescaling it
  image = image.resize((info.width, info.height), resample = Image.BILINEAR)

  # Draw ground truth boxes
  _draw_ground_truth_boxes(image = image, boxes = boxes)

  # Draw anchor boxes and intersection areas, if requested
  if draw_anchor_intersections or draw_anchor_points:
    _draw_anchor_box_intersections(image = image, ground_truth_boxes = boxes, draw_anchor_points = draw_anchor_points, input_image_shape = image_input_map.shape[1:])

  # Display image
  image.show()
  image.save("out_gt.png")

def show_proposed_regions(voc, filename, y_true, y_class, y_regression):
  # Load image and scale appropriately
  filepath = voc.get_full_path(filename = filename)
  info = voc.get_image_description(path = filepath)
  data = imageio.imread(filepath, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  image = image.resize((info.width, info.height), resample = Image.BILINEAR)
  ctx = ImageDraw.Draw(image, mode = "RGBA")

  # Get all anchors for this image size
  anchor_boxes, _ = region_proposal_network.compute_all_anchor_boxes(input_image_shape = (info.height, info.width, 3))

  boxes = []  # (y_min,, x_min, y_max, x_max)
  proposals = [] # (score, (y_min,x_min,y_max,x_max))
  for y in range(y_class.shape[1]):
    for x in range(y_class.shape[2]):
      for k in range(y_class.shape[3]):
        if y_class[0,y,x,k] > 0.5:# and y_true[0,y,x,k,1] > 0:# and y_true[0,y,x,k,0] > 0: # valid anchor and object
          # Extract predicted box
          anchor_box = anchor_boxes[y,x,k*4+0:k*4+4]
          box_params = y_regression[0,y,x,k*4+0:k*4+4]
          box = _convert_parameterized_box_to_points(box_params = box_params, anchor_center_y = anchor_box[0], anchor_center_x = anchor_box[1], anchor_height = anchor_box[2], anchor_width = anchor_box[3])
          boxes.append(box)
          proposals.append((y_class[0,y,x,k], box))
          # Draw filled anchor indicating region
          #draw_filled_rectangle(ctx = ctx, x_min = anchor_box[1] - 0.5 * anchor_box[3], x_max = anchor_box[1] + 0.5 * anchor_box[3], y_min = anchor_box[0] - 0.5 * anchor_box[2], y_max = anchor_box[0] + 0.5 * anchor_box[2], color = (0, 255, 0, 64))


  # Perform NMS on boxes
  print("initial proposals=%d" % len(proposals))
  final_proposals = []
  while len(proposals) > 0:
    # Extract best proposal
    best_score = -1
    best_idx = 0
    for i in range(len(proposals)):
      if proposals[i][0] > best_score:
        best_score = proposals[i][0]
        best_idx = i
    best_proposal = proposals[best_idx]
    final_proposals.append(best_proposal)
    del proposals[best_idx]
    
    # Compare IoU of current best against all remaining and discard those for which IoU is > 0.7
    proposals = [ proposal for proposal in proposals if intersection_over_union.intersection_over_union(box1=best_proposal[1], box2=proposal[1]) <= 0.7 ]
  print("final proposals=%d" % len(final_proposals))

  # Draw boxes
  for proposal in final_proposals:
    draw_rectangle(ctx = ctx, x_min = proposal[1][1], y_min = proposal[1][0], x_max = proposal[1][3], y_max = proposal[1][2], color = (255, 255, 0, 255), thickness = 1)
  #for box in boxes:
  #  draw_rectangle(ctx = ctx, x_min = box[1], y_min = box[0], x_max = box[3], y_max = box[2], color = (255, 255, 0, 255), thickness = 1)
  #image.show()
  image.save("out.png")

def _convert_parameterized_box_to_points(box_params, anchor_center_y, anchor_center_x, anchor_height, anchor_width):
  ty, tx, th, tw = box_params
  center_x = anchor_width * tx + anchor_center_x
  center_y = anchor_height * ty + anchor_center_y
  width = exp(tw) * anchor_width
  height = exp(th) * anchor_height
  y_min = center_y - 0.5 * height
  x_min = center_x - 0.5 * width
  y_max = center_y + 0.5 * height
  x_max = center_x + 0.5 * width
  return (y_min, x_min, y_max, x_max)

def _draw_ground_truth_boxes(image, boxes):
  # Draw green boxes
  ctx = ImageDraw.Draw(image)
  for box in boxes:
    draw_rectangle(ctx, x_min = box.x_min, y_min = box.y_min, x_max = box.x_max, y_max = box.y_max, color = (0, 255, 0, 255))
    print("box=%s" % str(box))

def _draw_anchor_box_intersections(image, ground_truth_boxes, draw_anchor_points, input_image_shape):
  ctx = ImageDraw.Draw(image, mode = "RGBA")

  anchor_boxes, anchor_boxes_valid = region_proposal_network.compute_all_anchor_boxes(input_image_shape = input_image_shape)

  ground_truth_regressions, positive_anchors, negative_anchors = region_proposal_network.compute_anchor_label_assignments(ground_truth_object_boxes = ground_truth_boxes, anchor_boxes = anchor_boxes, anchor_boxes_valid = anchor_boxes_valid)

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

        # Draw anchor center pos
        if draw_anchor_points:
          size = 1
          any_positive = True in [ ground_truth_regressions[y,x,k,2] > 0 for k in range(ground_truth_regressions.shape[2]) ]
          color = (0, 255, 0, 255) if any_positive else (255, 0, 0, 255)
          draw_filled_rectangle(ctx, y_min = center_y - size, x_min = center_x - size, y_max = center_y + size, x_max = center_x + size, color = color)

        ## Does it intersect with any ground truth box?
        #for box in ground_truth_boxes:
        #  box1 = (anchor_y_min, anchor_x_min, anchor_y_max, anchor_x_max)
        #  box2 = (box.y_min, box.x_min, box.y_max, box.x_max)

        #  iou = intersection_over_union.intersection_over_union(box1 = box1, box2 = box2)
        #  if iou > 0.7:
        #    # Render intersection area
        #    intersection_box = intersection_over_union.intersection(box1 = box1, box2 = box2)
        #    draw_filled_rectangle(ctx, y_min = intersection_box[0], x_min = intersection_box[1], y_max = intersection_box[2], x_max = intersection_box[3], color = (255, 0, 0, 64))

        #    # Draw the anchor box outline
        #    draw_rectangle(ctx, x_min = anchor_x_min, y_min = anchor_y_min, x_max = anchor_x_max, y_max = anchor_y_max, color = (255, 0, 0, 255))

  # Draw all anchor boxes labeled as object in yellow
  for i in range(len(positive_anchors)):
    y = positive_anchors[i][0]
    x = positive_anchors[i][1]
    k = positive_anchors[i][2]

    print("anchor=[%d,%d,%d]" % (y, x, k))

    # Extract box and draw
    box = anchor_boxes[y, x, k * 4 : k * 4 + 4]
    center_y = box[0]
    center_x = box[1]
    height = box[2]
    width = box[3]
    anchor_y_min = center_y - 0.5 * height
    anchor_x_min = center_x - 0.5 * width
    anchor_y_max = center_y + 0.5 * height
    anchor_x_max = center_x + 0.5 * width

    draw_rectangle(ctx, x_min = anchor_x_min, y_min = anchor_y_min, x_max = anchor_x_max, y_max = anchor_y_max, color = (255, 255, 0, 255))
