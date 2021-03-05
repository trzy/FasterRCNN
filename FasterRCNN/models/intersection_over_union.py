def intersection_over_union(box1, box2):
  intersection_area = _intersection_area(box1, box2)
  union_area = _union_area(box1, box2, intersection_area)
  epsilon = 1e-7
  return intersection_area / (union_area + epsilon)

def intersection(box1, box2):
  y1_min = box1[0]
  x1_min = box1[1]
  y1_max = box1[2]
  x1_max = box1[3]
  y2_min = box2[0]
  x2_min = box2[1]
  y2_max = box2[2]
  x2_max = box2[3]
  x1 = max(x1_min, x2_min)
  y1 = max(y1_min, y2_min)
  x2 = min(x1_max, x2_max)
  y2 = min(y1_max, y2_max)
  if x2 < x1 or y2 < y1:
    return None
  return (y1, x1, y2, x2)

def _intersection_area(box1, box2):
  y1_min = box1[0]
  x1_min = box1[1]
  y1_max = box1[2]
  x1_max = box1[3]
  y2_min = box2[0]
  x2_min = box2[1]
  y2_max = box2[2]
  x2_max = box2[3]
  x1 = max(x1_min, x2_min)
  y1 = max(y1_min, y2_min)
  x2 = min(x1_max, x2_max)
  y2 = min(y1_max, y2_max)
  if x2 < x1 or y2 < y1:
    return 0
  return (x2 - x1) * (y2 - y1)

def _union_area(box1, box2, intersection_area):
  y1_min = box1[0]
  x1_min = box1[1]
  y1_max = box1[2]
  x1_max = box1[3]
  y2_min = box2[0]
  x2_min = box2[1]
  y2_max = box2[2]
  x2_max = box2[3]  
  area_1 = (x1_max - x1_min) * (y1_max - y1_min)
  area_2 = (x2_max - x2_min) * (y2_max - y2_min)
  return area_1 + area_2 - intersection_area