## Description
This repository contains a Python 3 implementation of Collaborative/Contextual Probabilistic Movement Primitives (Contextual ProMPs) as described in [1], Section III.B. A regularized linear regression was used for conditioning. A detailed implementation as well as a use case can be found in [2]. A Python 2 implementation of a single DoF ProMP [3] was used as the code base [4].

## Usage example
``` python
# add demonstrations to ProMP containing one joint and one context
for demo_id in range(0, num_traj):
    pmp.add_demonstration(samples[demo_id])

# condition on new context and plot resulting trajectory
goal = np.zeros(2)
goal[1] = 0.1
pmp.set_goal(goal, sigma=1e-6)
generated_trajectory = pmp.generate_trajectory(sigma_noise)
plt.figure()
for joint_id, joint_name in enumerate(joints):
    print(joint_id)
    plt.plot(generated_trajectory[joint_id*num_points:(joint_id+1)*num_points, 0], label=joint_name)
plt.legend()
```
For more usage examples, please refer to `toyexample_promp.py`.
## References
[1] G. Maeda, M. Ewerton, R. Lioutikov, H. Ben Amor, J. Peters and G. Neumann, "Learning interaction for collaborative tasks with probabilistic movement primitives," 2014 IEEE-RAS International Conference on Humanoid Robots, Madrid, 2014, pp. 527-534.
[2] A. Sadybakasov, "Generalizing  to  New  Cup  Positions  in the  Game  of  Beer  Pong  Using  Contextual Probabilistic  Movement  Primitives", 2017 Report on Integrated Project: Robot Learning, TU Darmstadt, 2017
[3] A. Paraschos, G Neumann, C. Daniel, J. Peters, "Probabilistic movement primitives", Advances in NIPS, 2013
[4] https://github.com/baxter-flowers/promplib