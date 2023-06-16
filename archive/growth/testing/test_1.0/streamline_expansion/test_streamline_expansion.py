import numpy as np
import matplotlib.pyplot as plt

def streamline_extension(beta, dr=0.01):          
    y = ((beta**2) / 9) * ((27 * dr / (2 * beta**2) + 1) ** (2 / 3) - 1)
    x = np.sign(beta) * 2 * ((y**3 / beta**2) + (y / beta) ** 4) ** (1 / 2)
    return np.array([x, y])

def streamline_extension2(beta, dr=0.01):          
    y = np.empty(len(beta))
    mask = np.abs(beta) < 10
    y[mask] = ((beta[mask]**2) / 9) * ((27 * dr / (2 * beta[mask]**2) + 1) ** (2 / 3) - 1)
    y[~mask] = dr - (9*dr**2)/(4*beta[~mask]**2) + (27*dr**3) / \
            (2*beta[~mask]**4) - (1701*dr**4)/(16*beta[~mask]**6) # + (15309*dr**5)/(16*beta[~mask]**8)
    # x = np.around(np.sign(beta) * 2 * ((y**3 / beta**2) + (y / beta) ** 4) ** (1 / 2), 12)
    x = np.sign(beta) * 2 * ((y**3 / beta**2) + (y / beta) ** 4) ** (1 / 2)
    return np.array([x, y])
beta = np.arange(1,100,1)
plt.xscale('log')
(x, y) = streamline_extension(beta)
(x2, y2) = streamline_extension2(beta)
# plt.plot(beta,y)
# plt.plot(beta,y2)
plt.plot(beta,np.abs(y2-y))
plt.show()


beta = 1
dr = 0.01
y = ((beta**2) / 9) * ((27 * dr / (2 * beta**2) + 1) ** (2 / 3) - 1)
x = np.sign(beta) * 2 * ((y**3 / beta**2) + (y / beta) ** 4) ** (1 / 2)
print(x, y)


beta = 1
dr = 0.01
if np.abs(beta) < 1000:
    y = ((beta**2) / 9) * ((27 * dr / (2 * beta**2) + 1) ** (2 / 3) - 1)
else:
    y = dr - (9*dr**2)/(4*beta**2) + (27*dr**3) / \
        (2*beta**4) - (1701*dr**4)/(16*beta**6)
x = np.around(
    np.sign(beta) * 2 * ((y**3 / beta**2) +
                          (y / beta) ** 4) ** (1 / 2), 12)
print(x, y)