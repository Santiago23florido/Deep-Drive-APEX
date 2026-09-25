# LiDAR–IMU pose learning

Planar motion estimation for the APEX vehicle from a 2D LiDAR and a 6-axis
IMU. This folder contains, in chronological order:

1. the original single-track pipeline (`train_pose_fusion.py`);
2. a grid search of its three architectures on the multi-scenario dataset
   (`gridsearch_sqlite.py`).

The [references](#references) at the end cover the methods each model builds
on. Checkpoints, caches, CSV files and figures are written to
`learning/outputs/`, which is ignored by Git.

## 1. Original single-track pipeline (`train_pose_fusion.py`)

Self-supervised planar motion learning for one recorded track. The network
(`LidarImuPoseNet`) combines a local branch with the last two scans and a
causal context branch with five scans. Both share the spatial CNN but have
independent GRUs [13]; a learned gate decides how much of the contextual
correction to apply. The IMU samples cover the matching intervals. Outputs:

- `delta_pose = [dx, dy, dyaw]`;
- a diagonal uncertainty of that increment (heteroscedastic NLL [12]);
- estimated biases `[bax, bay, baz, bgx, bgy, bgz]`.

The translation includes a differentiable recurrent mechanization:

```text
v[t]  = v[t-1] + (a[t] - bias_a[t]) dt + delta_v_net[t]
dp[t] = v[t-1] dt + 0.5 (a[t] - bias_a[t]) dt^2 + delta_p_net[t]
```

The GRU initializes the velocity of each window and only learns the residual
corrections. The rotation uses the gyro integral corrected by the estimated
bias in the same way.

Readings are synchronized by their real timestamps: every interval between
two scans keeps the IMU samples that actually arrived, without resampling
them to a fixed count. Batches are padded to the longest interval and the
valid length goes to a packed GRU, which ignores the padding. The LiDAR tensor
keeps the 360 beams of each of the five context scans.

The LiDAR correction is

```text
delta_pose = delta_pair + sigmoid(gate) * delta_context + delta_inertial
```

Besides the per-step losses, the composition of the six increments of a
sequence must match the composition of the sensor-only increments, so that
the two branches cannot produce local motions that contradict the long
context.

The exact Gazebo pose **is not used for training nor for checkpoint
selection** in the default mode. Training uses the relative motion computed
from the LiDAR alone with a point-to-point ICP written in PyTorch
(`torch.cdist` and `torch.linalg.svd`, [1], [20]), consistency with the gyro
integral, temporal smoothness and bias regularization. The ground truth is
opened once at the end to evaluate and plot the 15 % validation split; the
15 % test split stays reserved and only its self-supervised loss is reported.

The optional `--supervised-ground-truth` mode replaces those ICP targets with
relative increments from the exact pose, to measure the supervised ceiling
and study generalization across tracks and sensors.

The split is chronological so that overlapping windows of one lap never fall
on both sides of a partition: 70 % train, 15 % validation, 15 % test.

```bash
python3 learning/lidar_imu_pose/train_pose_fusion.py \
  --run-dir simulation/data/fusion_research/sensor_lap_final_20260924_173000 \
  --epochs 30 --device cuda
```

Training directly against the exact simulator pose, with learning curves:

```bash
python3 learning/lidar_imu_pose/train_pose_fusion.py \
  --run-dir simulation/data/fusion_research/sensor_lap_final_20260924_173000 \
  --output-dir simulation/learning/outputs/lidar_imu_pose_supervised \
  --epochs 40 --device cuda --supervised-ground-truth
```

Checkpoints, caches, metrics and figures go to
`learning/outputs/lidar_imu_pose/` by default.

## References

1. P. J. Besl, N. D. McKay, "A method for registration of 3-D shapes", IEEE TPAMI 14(2), 1992.
2. Y. Chen, G. Medioni, "Object modelling by registration of multiple range images", Image and Vision Computing 10(3), 1992.
3. A. Censi, "An ICP variant using a point-to-line metric", IEEE ICRA 2008.
4. J. Zhang, S. Singh, "LOAM: Lidar Odometry and Mapping in Real-time", RSS 2014.
5. I. Vizzo, T. Guadagnino, B. Mersch, L. Wiesmann, J. Behley, C. Stachniss, "KISS-ICP: In Defense of Point-to-Point ICP – Simple, Accurate, and Robust Registration If Done the Right Way", IEEE RA-L 8(2), 2023.
6. J. Zhang, M. Kaess, S. Singh, "On degeneracy of optimization-based state estimation problems", IEEE ICRA 2016.
7. Z. Zhang, "Parameter estimation techniques: a tutorial with application to conic fitting", Image and Vision Computing 15(1), 1997.
8. W. Hess, D. Kohler, H. Rapp, D. Andor, "Real-time loop closure in 2D LIDAR SLAM", IEEE ICRA 2016.
9. P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems", 2nd ed., Artech House, 2013.
10. I. Skog, P. Händel, J.-O. Nilsson, J. Rantakokko, "Zero-velocity detection — An algorithm evaluation", IEEE Trans. Biomedical Engineering 57(11), 2010.
11. G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, Y. C. Eldar, "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics", IEEE Trans. Signal Processing 70, 2022.
12. A. Kendall, Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017.
13. K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation", EMNLP 2014.
14. R. J. Williams, J. Peng, "An efficient gradient-based algorithm for on-line training of recurrent network trajectories", Neural Computation 2(4), 1990.
15. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 15, 2014.
16. I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization", ICLR 2019.
17. L. N. Smith, N. Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates", Proc. SPIE 11006, 2019.
18. A. Geiger, P. Lenz, R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite", CVPR 2012.
19. J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, "A benchmark for the evaluation of RGB-D SLAM systems", IROS 2012.
20. K. S. Arun, T. S. Huang, S. D. Blostein, "Least-squares fitting of two 3-D point sets", IEEE TPAMI 9(5), 1987.
21. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019.
