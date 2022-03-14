import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy

mu = 0
steps_num = 1500
traj_num = 1500

sigma_vs_deltafraq = [[], []]

for sigma in np.linspace(0, 2, 500):
    generated_steps = [np.zeros(traj_num)]

    for step in range(steps_num):
        generated_steps.append(generated_steps[step] + np.random.normal(mu, sigma, traj_num))

    num_of_cross_traj = 0
    trajectories = []
    for trajectory in range(traj_num):
        is_larger = 0
        trajectories.append([])
        for step in range(len(generated_steps)):
            trajectories[trajectory].append(generated_steps[step][trajectory])
            if generated_steps[step][trajectory] >= 1.69:
                is_larger += 1
        if is_larger != 0:
            num_of_cross_traj += 1
        # plt.plot(range(steps_num+1), trajectories[trajectory])

    sigma_vs_deltafraq[0].append(sigma)
    sigma_vs_deltafraq[1].append(num_of_cross_traj / traj_num)


def func(z, a, b):
    return 1 - scipy.special.erf(a / np.sqrt(z)) + b


popt, pcov = curve_fit(func, sigma_vs_deltafraq[0], sigma_vs_deltafraq[1])

plt.plot(sigma_vs_deltafraq[0], sigma_vs_deltafraq[1], label='simulation')
plt.plot(sigma_vs_deltafraq[0], func(sigma_vs_deltafraq[1], *popt), 'g--',
         label=r'fit: erfc($\frac{a}{\sqrt{S}}$)+b ; ' + 'a=%5.3f, b=%5.3f' % tuple(popt))
plt.xlabel('S')
plt.ylabel('F(S)')
plt.legend()
plt.show()

# plt.plot(range(steps_num+1), 1.69*np.ones(steps_num+1),'--',label="$\delta_c$")
# plt.xlabel('steps')
# plt.ylabel('$\delta$')
# plt.legend()
# plt.show()
