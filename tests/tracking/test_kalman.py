from __future__ import annotations

from celikkubbe.tracking.kalman import (
    RANGE_MEASUREMENT_NOISE_DEPTH,
    RANGE_MEASUREMENT_NOISE_SIZE,
    CentroidKalmanFilter,
    RangeKalmanFilter,
    measurement_noise_for_source,
)


def test_centroid_filter_predicts_along_constant_velocity() -> None:
    kf = CentroidKalmanFilter(0.5, 0.5)
    kf.update(0.51, 0.50)
    kf.predict(dt=1.0)
    kf.update(0.52, 0.50)
    for _ in range(20):
        kf.predict(dt=1.0)
        x, y = kf.position
        kf.update(x + 0.01, y)

    x_before, _ = kf.position
    kf.predict(dt=1.0)
    x_after, _ = kf.position
    assert x_after > x_before


def test_centroid_filter_velocity_converges_towards_true_speed() -> None:
    kf = CentroidKalmanFilter(0.0, 0.5)
    true_speed = 0.02
    x = 0.0
    for _ in range(30):
        kf.predict(dt=1.0)
        x += true_speed
        kf.update(x, 0.5)
    vx, vy = kf.velocity
    assert abs(vx - true_speed) < 0.005
    assert abs(vy) < 0.005


def test_measurement_noise_depth_is_lower_than_size() -> None:
    assert RANGE_MEASUREMENT_NOISE_DEPTH < RANGE_MEASUREMENT_NOISE_SIZE
    assert measurement_noise_for_source("depth") == RANGE_MEASUREMENT_NOISE_DEPTH
    assert measurement_noise_for_source("size") == RANGE_MEASUREMENT_NOISE_SIZE
    assert measurement_noise_for_source("none") == RANGE_MEASUREMENT_NOISE_SIZE


def test_range_filter_idles_without_update() -> None:
    rf = RangeKalmanFilter(10.0, RANGE_MEASUREMENT_NOISE_DEPTH)
    rf.predict()
    rf.predict()
    rf.predict()
    assert abs(rf.value - 10.0) < 0.5


def test_range_filter_converges_faster_with_low_noise_depth_measurements() -> None:
    depth = RangeKalmanFilter(5.0, RANGE_MEASUREMENT_NOISE_DEPTH)
    size = RangeKalmanFilter(5.0, RANGE_MEASUREMENT_NOISE_SIZE)
    for _ in range(5):
        depth.predict()
        depth.update(8.0, RANGE_MEASUREMENT_NOISE_DEPTH)
        size.predict()
        size.update(8.0, RANGE_MEASUREMENT_NOISE_SIZE)

    assert abs(depth.value - 8.0) < abs(size.value - 8.0)
