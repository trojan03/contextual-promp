import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy.linalg import inv

class ProMPContext(object):
    def __init__(self, joints, num_basis=20, sigma=0.1, num_points=1750):
        self.joints = joints
        self.z = np.linspace(0, 1, num_points)
        self.num_points = len(self.z)
        self.num_basis = num_basis
        self.sigma = sigma
        self.num_joints = len(self.joints)
        self.no_traj = True
        self.num_points = num_points
        self.unconditioned_trajectory = []
        self.generated_trajectory = []

        # basis functions matrix
        self.centers = np.arange(0,self.num_basis )/(self.num_basis - 1.0)
        self.phi = np.exp(-0.5 * (np.array(list(map(lambda x: x - self.centers, np.tile(self.z, (self.num_basis, 1)).T))).T ** 2
                                  / self.sigma**2))
        self.phi /= np.sum(self.phi, axis=0)
        self.Phi = np.zeros((self.num_basis*self.num_joints, num_points*self.num_joints))
        for i in range(0, self.num_joints):
            self.Phi[i*self.num_basis:(i+1)*self.num_basis, i*num_points:(i+1)*num_points] = self.phi

        self.viapoints = []
        self.W = []
        self.nrTraj = 0
        self.meanW = None
        self.sigmaW = None
        self.Y = []
        self.sample = []

    def add_demonstration(self, demonstration):
        self.nrTraj += 1
        currentY = []
        currentW = []

        for joint in range(0, self.num_joints):
            interpolate = interp1d(np.linspace(0, 1, len(demonstration[:, joint])), demonstration[:, joint], kind='cubic')
            stretched_demo = interpolate(self.z)
            currentY.append(stretched_demo)
            aux = np.dot(self.phi, self.phi.T)
            currentJointW = np.dot(np.linalg.inv(aux + np.eye(aux.shape[1])*1e-6), np.dot(self.phi, np.array(stretched_demo).T))
            currentW = np.append(currentW, currentJointW)

        if self.no_traj:
            self.Y = np.array(currentY)
            self.W = currentW
            self.no_traj = False
            self.meanW = self.W
        else:
            self.Y = np.vstack((self.Y, np.array(currentY)))
            self.W = np.vstack((self.W, currentW.T))
            self.meanW = np.mean(self.W, 0)

        self.sigmaW = np.cov(self.W.T)

    def get_trajectory_fromweights(self, weights):
        aux = np.dot(self.Phi.T, weights)
        traj = []
        for joint_id, joint_name in enumerate(self.joints):
            traj.append(aux[joint_id*self.num_points:(joint_id+1)*self.num_points])
        return traj

    def plot_unconditioned_joints(self):
        sampW = np.random.multivariate_normal(self.meanW, self.sigmaW, 1).T
        sample = np.dot(self.Phi.T, sampW)
        std = self.get_std()

        plt.figure(figsize=(6, 4))
        for joint_id, joint_name in enumerate(self.joints):
            if 'joint' in joint_name:
                plt.plot(np.arange(0, len(sample[joint_id*self.num_points:(joint_id+1)*self.num_points, 0])) /
                         self.num_points, sample[joint_id*self.num_points:(joint_id+1)*self.num_points, 0], label=joint_name)
            else:
                plt.plot(np.arange(0, len(sample[joint_id*self.num_points:(joint_id+1)*self.num_points, 0])) /
                         self.num_points, sample[joint_id*self.num_points:(joint_id+1)*self.num_points, 0], '-.', label=joint_name)
            plt.fill_between(np.arange(0, len(sample[joint_id*self.num_points:(joint_id+1)*self.num_points, 0])) /
                             self.num_points, sample[joint_id*self.num_points:(joint_id+1)*self.num_points, 0] -
                             std[joint_id*self.num_points:(joint_id+1)*self.num_points],
                                 sample[joint_id*self.num_points:(joint_id+1)*self.num_points, 0] +
                             std[joint_id*self.num_points:(joint_id+1)*self.num_points],
                                 alpha=0.2)
        plt.xlabel('t [s]')
        plt.ylabel('joint position [rad]')
        plt.legend()

    def generate_trajectory(self, sigma=1e-6):
        newMu = self.meanW
        newSigma = self.sigmaW

        for viapoint in self.viapoints:
            phiT = np.exp(-.5 * (np.array(list(map(lambda x: x - self.centers, np.tile(viapoint['t'], (self.num_basis, 1)).T))).T ** 2
                                 / (self.sigma ** 2)))
            phiT = phiT / sum(phiT)
            PhiT = np.zeros((self.num_basis *self.num_joints, self.num_joints))

            for i in range(0, self.num_joints):
                PhiT[i*self.num_basis:(i+1)*self.num_basis, i:(i+1)] = phiT

            ######### Conditioning
            aux = viapoint['sigma'] + np.dot(np.dot(PhiT.T, newSigma), PhiT)

            # trick: add some noise to avoid instabilities of matrix inverse
            newMu = newMu + np.dot(np.dot(np.dot(newSigma, PhiT), inv(aux + np.eye(aux.shape[1])*sigma)),
                                   viapoint['trajectory'] - np.dot(PhiT.T, newMu))

            newSigma = newSigma - np.dot(np.dot(np.dot(newSigma, PhiT), inv(aux + np.eye(aux.shape[1])*sigma)),
                                         np.dot(PhiT.T, newSigma))
            ##########

        sampW = np.random.multivariate_normal(newMu, newSigma, 1).T
        return np.dot(self.Phi.T, sampW)

    def set_goal(self, trajectory, sigma=1e-6):
        if len(trajectory) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(trajectory), self.num_joints))
        else:
            self.add_viapoint(1., trajectory, sigma)

    def set_start(self, trajectory, sigma=1e-6):
        if len(trajectory) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(trajectory), self.num_joints))
        else:
            self.add_viapoint(0., trajectory, sigma)

    def add_viapoint(self, t, trajectory, sigma=1e-6):
        if len(trajectory) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(trajectory), self.num_joints))
        else:
            self.viapoints.append({"t": t, "trajectory": trajectory, "sigma": sigma})

    def get_bounds(self, t):
        mean = self._get_mean(t)
        std = self.get_std()
        std_t = []
        for joint_id, joint_name in enumerate(self.joints):
            std_t.append(std[joint_id*t])
        return mean - std_t, mean + std_t

    def get_mean(self, t):
        mean = np.dot(self.Phi.T, self.meanW)
        meanT = []
        for joint_id, joint_name in enumerate(self.joints):
            meanT.append(mean[joint_id*self.num_points:(joint_id+1)*self.num_points])
        return np.array(meanT)[:, t]

    def get_std(self):
        std = 2*np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW, self.Phi))))
        return std

    def clear_viapoints(self):
        del self.viapoints[:]

    def plot_conditioned_joints(self):
        generated = self.generate_trajectory()
        plt.figure(figsize=(6, 4))
        for joint_id, joint_name in enumerate(self.joints):
            if 'joint' in joint_name:
                plt.plot(np.arange(0, len(generated[joint_id*self.num_points:(joint_id+1)*self.num_points, 0])) /
                         self.num_points, generated[joint_id*self.num_points:(joint_id+1)*self.num_points, 0], label=joint_name)
            else:
                plt.plot(np.arange(0, len(generated[joint_id*self.num_points:(joint_id+1)*self.num_points, 0])) /
                         self.num_points, generated[joint_id*self.num_points:(joint_id+1)*self.num_points, 0], '-.', label=joint_name)
        plt.xlabel('t [s]')
        plt.ylabel('joint position [rad]')
        plt.legend()

    def get_conditioned_trajectory(self):
        generated = self.generate_trajectory()
        qs = []
        for joint_id, joint_name in enumerate(self.joints):
            qs.append(generated[joint_id*self.num_points:(joint_id+1)*self.num_points])
        return np.array(qs)[:, :, 0]