import numpy as np
from contextual_promp.contextual_promp import *

num_traj = 20
sigma_noise=0.03
x = np.arange(0,1,0.01)
A = np.array([.2, .2, .01, -.05])
X = np.vstack((np.sin(5*x), x**2, x, np.ones((1,len(x)))))

# we use one joint and one context
# feel free to use more joints or contexts
joints = ["joint_x", "context"]

Y = np.zeros((num_traj, len(x)))
samples = []
for traj in range(0, num_traj):
    sample = np.dot(A + sigma_noise * np.random.randn(1,4), X)[0]
    sample = np.vstack((sample, np.tile(sigma_noise * np.random.randn() + 0.15, len(x))))
    samples.append(sample)

samples = np.array(samples)
num_points=len(x)
plt.figure(figsize=(6, 4))
for i in range(0, num_traj):
    if i == 0:
        plt.plot(np.arange(0, len(samples[i, 0, :].T)) / num_points, samples[i, 0, :].T, color='blue', label='joint_x')
        plt.plot(np.arange(0, len(samples[i, 1, :].T)) / num_points, samples[i, 1, :].T, '-.', color='orange', label='context')
    else:
        plt.plot(np.arange(0, len(samples[i, 0, :].T)) / num_points, samples[i, 0, :].T, color='blue')
        plt.plot(np.arange(0, len(samples[i, 1, :].T)) / num_points, samples[i, 1, :].T, '-.', color='orange')
plt.xlabel('t [s]')
plt.ylabel('joint position [rad]')
plt.legend()

pmp = ProMPContext(joints, num_points=num_points)

# add demonstrations to ProMP
for demo_id in range(0, num_traj):
    pmp.add_demonstration(samples[demo_id, :, :].T)

# plot mean and standard deviation of demonstrations
pmp.plot_unconditioned_joints()

# condition on context=0.1
goal = np.zeros(2)
goal[1] = 0.1
pmp.clear_viapoints()
pmp.set_goal(goal, sigma=1e-6)
pmp.plot_conditioned_joints()

# alternatively
# goal = np.zeros(2)
# goal[1] = 0.1
# pmp.clear_viapoints()
# pmp.set_goal(goal, sigma=1e-6)
# generated_trajectory = pmp.generate_trajectory(sigma_noise)
# plt.figure()
# for joint_id, joint_name in enumerate(joints):
#     print(joint_id)
#     plt.plot(generated_trajectory[joint_id*num_points:(joint_id+1)*num_points, 0], label=joint_name)
# plt.legend()

