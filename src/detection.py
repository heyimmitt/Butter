from ultralytics import YOLO
from density_scorer import load_lane_polygons, get_all_lane_scores

model = YOLO('data/models/yolov8n.pt')
lane_polygons = load_lane_polygons()


def get_frame_scores(source, classes=[2, 3, 5, 7], conf=0.5):
    """Runs detection on a frame/video and returns per-lane congestion scores."""
    total = 0
    frame_count = 0
    results = model(source, classes=classes, conf=conf)

    all_scores = []
    for frame_result in results:
        boxes = frame_result.boxes.xyxy.cpu().numpy().tolist()
        scores = get_all_lane_scores(boxes, lane_polygons)
        all_scores.append(scores)
        total += sum(scores.values())
        frame_count += 1

    return total/frame_count


    


if __name__ == "__main__":
    # lets you still run `python src/detection.py` directly for a quick manual test
    avg = get_frame_scores('data/samples/test_clip.webm')
    print(f"avg is {avg}")