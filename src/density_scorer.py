"""
src/density.py

Turns YOLO detections into a per-lane congestion score (0-1) each frame.
"""

import json
from shapely.geometry import Point, Polygon


def load_lane_polygons(config_path="config/lanes.json"):
    """Load lane ROI polygons from the JSON config file.
    Returns something like {"north": [[x,y], [x,y], ...], ...}
    """
    with open(config_path) as f:
        raw = json.load(f)
    # Convert each lane's point list into an actual Shapely Polygon object
    # so we can reuse it for both point-in-polygon checks and area calculation.
    return {lane: Polygon(points) for lane, points in raw.items()}


def get_centroid(box):
    """box = (x1, y1, x2, y2) -> returns (center_x, center_y)"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_box_area(box):
    """box = (x1, y1, x2, y2) -> width * height"""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def assign_to_lanes(detections, lane_polygons):
    """
    detections: list of boxes, each (x1, y1, x2, y2)
    lane_polygons: dict of {lane_name: Shapely Polygon}

    Returns {lane_name: [box, box, ...]} — boxes whose centroid
    doesn't fall inside ANY lane polygon are silently dropped.
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


def compute_congestion_score(lane_boxes, lane_polygon, max_expected_count=12,
                               occupancy_weight=0.5, count_weight=0.5):
    """
    lane_boxes: list of boxes assigned to this lane (from assign_to_lanes)
    lane_polygon: this lane's Shapely Polygon (for area comparison)
    max_expected_count: assumed "lane is basically full" vehicle count —
                         tune this per lane based on its real length/size
    occupancy_weight / count_weight: how much each raw signal contributes
                                       to the final blended score (should sum to 1)

    Returns a single float 0-1 representing how congested this lane is.
    """
    count = len(lane_boxes)
    count_ratio = min(count / max_expected_count, 1.0)

    occupied_area = sum(get_box_area(box) for box in lane_boxes)
    lane_area = lane_polygon.area
    occupancy_ratio = min(occupied_area / lane_area, 1.0) if lane_area > 0 else 0.0

    score = (occupancy_weight * occupancy_ratio) + (count_weight * count_ratio)
    return round(score, 4)


def get_all_lane_scores(detections, lane_polygons, max_expected_count=12):
    """
    Top-level function — call this once per frame.

    detections: list of boxes from YOLO for this frame
    lane_polygons: dict of {lane_name: Shapely Polygon} from load_lane_polygons()

    Returns {"north": 0.42, "south": 0.31, ...} — ready to feed the fuzzy engine.
    """
    assigned = assign_to_lanes(detections, lane_polygons)

    scores = {}
    for lane_name, boxes in assigned.items():
        scores[lane_name] = compute_congestion_score(
            boxes, lane_polygons[lane_name], max_expected_count
        )

    return scores


if __name__ == "__main__":
    # Quick manual sanity check — replace with a real YOLO output when ready
    lane_polygons = load_lane_polygons()
    fake_detections = [
        (60, 80, 150, 140),   # a box roughly in your marked lane
    ]
    print(get_all_lane_scores(fake_detections, lane_polygons))