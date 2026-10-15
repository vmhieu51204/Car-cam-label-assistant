"""Manual pixel-region trajectories, interpolated on the original frame timeline."""

import math


def expand_tracks(tracks: list[dict], times: list[float], width: int, height: int) -> list[dict]:
    result = []
    for track in tracks:
        keys = track["keyframes"]
        frames = [k["frame"] for k in keys]
        if frames != sorted(set(frames)) or not frames or not 0 <= frames[0] <= frames[-1] < len(times):
            raise ValueError("Privacy track keyframes must be unique, ordered and inside the video")
        for key in keys:
            x1, y1, x2, y2 = key["box"]
            if not all(math.isfinite(x) for x in key["box"]) or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                raise ValueError("Privacy track rectangle outside image")
        for i, key in enumerate(keys):
            end = keys[i + 1]["frame"] if i + 1 < len(keys) else key["frame"]
            following = keys[i + 1] if i + 1 < len(keys) else key
            for frame in range(key["frame"], end + (i == len(keys) - 1)):
                weight = (
                    (times[frame] - times[key["frame"]]) / (times[end] - times[key["frame"]])
                    if end > key["frame"]
                    else 0
                )
                box = [a + weight * (b - a) for a, b in zip(key["box"], following["box"], strict=True)]
                result.append({"start": frame, "end": frame, "box": box, "track_id": track["track_id"]})
    return result
