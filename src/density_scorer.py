'''
1. load lane config
    def load_lane_polygons(json_config_file)
    read lanes.json 

2. get a box's centroid
    def get_centroid(box)

3. check if point is in polygon
    def is_in_lane(point, polygon)
    check if the point (centroid of car) is in polygon (lane) using ray casting algorithm
    and function like matplotlib.path.Path(polygon).contains_point(point) 
    or shapely.geometry.Point(x,y).within(Polygon(polygon)) <- using this

4. get detections and assign all to lanes
    def assign_to_lane(detections, lane_polygons)
    input: yolo detection results for a frame -- list of boxes plus our loaded lane polygons
    logic: for each box, get its centroid, check which lane the box belongs to using is_in_lane()
    and append the box to the appropriate lane

5. calculate polygon area (for occupancy %age)
    def polygon_area(polygon)
    use shapely and Polygon(coords).area

6. compute actual congestion score
    def compute_congestion_score(lane_boxes, lane_polygon, max_expected_count=12)
    count = len(lane_boxes)
    count_ratio = min(count / max_expected_count, 1.0) -- capped at 1
    occupied_area = sum of (x2-x1)*(y2-y1) for each box
    occupancy_ratio = min(occupied_area / polygon_area(lane_polygon), 1.0) 
    -- also capped, in case boxes overlap and sum past the polygon's actual area
    final score: pick blend weights, e.g. score = 0.5 * occupancy_ratio + 0.5 * count_ratio 
'''


import json
from shapely.geometry import Point, Polygon


def load_lane_polygons(config_path="config/lanes.json"):
    """
    returns something like {"north": [[x,y], [x,y], ...], ...}
    """
    with open(config_path) as f:
        raw = json.load(f)
    # Convert each lane's point list into an actual Shapely Polygon object
    # so we can reuse it for both point-in-polygon checks and area calculation.
    return {lane: Polygon(points) for lane, points in raw.items()}


def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def assign_to_lanes(detections, lane_polygons):
    """
    detections: list of boxes, each (x1, y1, x2, y2)
    lane_polygons: dict of {lane_name: Shapely Polygon}

    Returns {lane_name: [box, box, ...]} 
    boxes whose centroid doesn't fall inside ANY lane polygon are silently dropped.
    """
    assigned = {lane: [] for lane in lane_polygons}

    for box in detections:
        cx, cy = get_centroid(box)
        point = Point(cx, cy)

        for lane_name, polygon in lane_polygons.items():
            if polygon.contains(point):
                assigned[lane_name].append(box)
                break  # a box can only belong to one lane, stop checking once matched

    return assigned


def compute_congestion_score(lane_boxes, lane_polygon, max_expected_count=5, occupancy_weight=0.5, count_weight=0.5):
    """
    lane_boxes: list of boxes assigned to this lane (from assign_to_lanes)
    lane_polygon: this lane's Shapely Polygon (for area comparison)
    max_expected_count: assumed "lane is basically full" vehicle count -- tune this per lane based on its real length/size
    occupancy_weight / count_weight: how much each raw signal contributes to the final blended score (should sum to 1)

    Returns a single float 0-1 representing how congested this lane is.
    """
    count = len(lane_boxes)
    count_ratio = min(count / max_expected_count, 1.0)

    occupied_area = sum(get_box_area(box) for box in lane_boxes)
    lane_area = lane_polygon.area
    occupancy_ratio = min(occupied_area / lane_area, 1.0) if lane_area > 0 else 0.0

    score = (occupancy_weight * occupancy_ratio) + (count_weight * count_ratio)
    return round(score, 4)


def get_all_lane_scores(detections, lane_polygons, max_expected_count = 5):
    """
    Top-level function — call this once per frame.

    detections: list of boxes from YOLO for this frame
    lane_polygons: dict of {lane_name: Shapely Polygon} from load_lane_polygons()

    Returns {"north": 0.42, "south": 0.31, ...} — ready to feed the fuzzy engine.
    """
    assigned = assign_to_lanes(detections, lane_polygons)

    scores = {}
    for lane_name, boxes in assigned.items():
        scores[lane_name] = compute_congestion_score(boxes, lane_polygons[lane_name], max_expected_count)

    return scores


if __name__ == "__main__":
    # Quick manual sanity check — replace with a real YOLO output when ready
    lane_polygons = load_lane_polygons()
    fake_detections = [
        (536, 325, 575, 447),   # test box
    ]
    print(get_all_lane_scores(fake_detections, lane_polygons))