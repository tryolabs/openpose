# flake8: noqa
# As implemented in https://github.com/abewley/sort but with some modifications

import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanPoseTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, pose, tracked_points_num):
    """
    Initialises a tracker using initial bounding box.
    """
    dim_x = 2 * 2 * tracked_points_num  # We need to accomodate for velocities
    dim_z = 2 * tracked_points_num  # We need to accomodate for velocities
    self.dim_z = dim_z
    # Define constant velocity model
    self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # State transition matrix (models physics): numpy.array()
    self.kf.F = np.eye(dim_x)
    dt = 1  # At each step we update pos with v * dt
    # Add position vs velocity update rules
    for p in range(dim_z):
        self.kf.F[p, p + dim_z] = dt
    # Process uncertainty: numpy.array(dim_x, dim_x)
    self.kf.Q[-1,-1] *= 0.01  # TODO check
    self.kf.Q[dim_z:, dim_z:] *= 0.01

    # Measurement function: numpy.array(dim_z, dim_x)
    self.kf.H = np.eye(dim_z, dim_x,)
    # Measurement uncertainty (sensor noise): numpy.array(dim_z, dim_z)
    # NOTE: Reduced from 10 to 1 as it made our predictions lag behind our detections too much
    self.kf.R *= 1.

    # Initial state: numpy.array(dim_x, 1)
    self.kf.x[:dim_z] = self.convert_to_kf_x_format(pose)
    # Initial state covariance matrix: numpy.array(dim_x, dim_x)
    self.kf.P[dim_z:, dim_z:] *= 1000. # Give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.  # TODO we get 10 * 1000 in velocities, is this correct?

    self.time_since_update = 0
    self.id = KalmanPoseTracker.count
    KalmanPoseTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.debug_dict = {}
    # # NOTE: I am overriding our predictions by setting this huge K because when a limb dissapears
    # #       the kalman filter interpolates between its position and (0, 0) and fucks my code up.
    # self.kf.K += 10000

    self.last_detection = pose


  def update(self, pose, debug_dict, R=None, H=None):
    """
    Updates the state vector with observed pose.
    """
    # pose[pose == 0] = np.nan
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    # TODO Isn't there a better way to do this than just not updating it?
    # At least tell the filter that a time step happened?
    if pose is not None:
      self.last_detection = pose
      self.kf.update(self.convert_to_kf_x_format(pose), R, H)
      self.debug_dict.update(debug_dict)


  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.unconvert_to_kf_x_format(self.kf.x))
    return self.history[-1]


  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.unconvert_to_kf_x_format(self.kf.x)


  def convert_to_kf_x_format(self, pose):
      return np.expand_dims(pose.flatten(), 0).T
  
  
  def unconvert_to_kf_x_format(self, pose):
      return pose.T.flatten()[:self.dim_z].reshape(-1, 2)
