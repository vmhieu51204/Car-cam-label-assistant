from scripts.eval_autolabel import Box, tracking_metrics


def test_fragmentation_counts_reacquisition_but_not_gaps_in_ground_truth():
    gt = [Box(i, "Car", (0, 0, 10, 10), "gt") for i in range(4)]
    pred = [Box(i, "Car", (0, 0, 10, 10), "a" if i == 0 else "b") for i in (0, 2, 3)]
    metrics = tracking_metrics(gt, pred)
    assert metrics["id_switches"] == 1 and metrics["fragmentations"] == 1
    assert tracking_metrics([g for g in gt if g.frame != 1], pred)["fragmentations"] == 0
