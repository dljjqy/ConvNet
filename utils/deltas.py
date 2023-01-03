import numpy as np
from scipy.stats import multivariate_normal

def normal(x, y, h, mean=[0, 0]):
    var = np.diag([1] * 2) * h**2
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, var)
    return rv.pdf(pos)

def normal_fourth(x, y, h, a=1):
    m1 = [-a/2, -a/2]
    m2 = [a/2, a/2]
    m3 = [-a/2, a/2]
    m4 = [a/2, -a/2]
    var = np.diag([1] * 2) * h**2
    pos = np.dstack((x, y))
    rv1 = multivariate_normal(m1, var)
    rv2 = multivariate_normal(m2, var)
    rv3 = multivariate_normal(m3, var)
    rv4 = multivariate_normal(m4, var)
    return rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos) + rv4.pdf(pos)

def R(x, y):
    return np.sqrt(x**2 + y**2)

def yita11_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask * ((6 * (3 - 4 * r)/np.pi)/H2)

def yita12_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask * ((12 * (5 * r**2 - 8 * r + 3)/np.pi)/H2)

def yita22_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((12 * (15 * r**2 - 20 * r + 6)/np.pi)/H2)

def yita23_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((-60 * (7 * r**3 - 15 * r**2 + 10 * r - 2)/np.pi)/H2)

def yita25_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((84*(24*r**5-70*r**4+70*r**3-25*r**2+1)/np.pi)/H2)

def yita2cos_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*(((-1 / (9*np.pi**4-104*np.pi**2+48)) *
                  ((81*np.pi/16)*(3*np.pi**4-32*np.pi**2+48)*np.cos(3*np.pi*r) +
                   2*np.pi*(9*np.pi**4-80*np.pi**2+48) * np.cos(2*np.pi*r) +
                   ((np.pi/16)*(45*np.pi**2+32*np.pi**2-48)*np.cos(np.pi*r) + 144*np.pi)))/H2)
