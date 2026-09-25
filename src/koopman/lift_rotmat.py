import numpy as np
from itertools import combinations

def quat_to_rotmat(q):
    q1, q2, q3, q4 = q[:,0], q[:,1], q[:,2], q[:,3]
    R = np.zeros((len(q), 3, 3))
    R[:,0,0] = 1 - 2*(q2**2 + q3**2)
    R[:,0,1] = 2*(q1*q2 - q3*q4)
    R[:,0,2] = 2*(q1*q3 + q2*q4)
    R[:,1,0] = 2*(q1*q2 + q3*q4)
    R[:,1,1] = 1 - 2*(q1**2 + q3**2)
    R[:,1,2] = 2*(q2*q3 - q1*q4)
    R[:,2,0] = 2*(q1*q3 - q2*q4)
    R[:,2,1] = 2*(q2*q3 + q1*q4)
    R[:,2,2] = 1 - 2*(q1**2 + q2**2)
    return R

def lift_rotmat(states_16):
    # states_16 columns: 0-2 pos, 3-6 quat, 7-9 rpy, 10-12 vel, 13-15 angvel
    pos = states_16[:, 0:3]
    vel = states_16[:, 10:13]
    angvel = states_16[:, 13:16]
    q = states_16[:, 3:7]

    R = quat_to_rotmat(q)  # (N, 3, 3)
    R_flat = R.reshape(len(states_16), 9)  # flatten 3x3 into 9 numbers

    # R @ omega (rotation matrix applied to angular velocity)
    R_omega = np.einsum('nij,nj->ni', R, angvel)  # (N, 3)
    # R @ omega^2 (elementwise square of angular velocity, then rotated)
    R_omega_sq = np.einsum('nij,nj->ni', R, angvel**2)  # (N, 3)

    return np.concatenate([pos, vel, R_flat, R_omega, R_omega_sq], axis=1)

if __name__ == "__main__":
    d = np.load('data/nominal/flight_log.npz')
    states = d['states'][:, :16]
    lifted = lift_rotmat(states)
    print(f"Lifted shape: {lifted.shape}")
    print(f"Sample lifted vector at step 0: {lifted[0]}")
