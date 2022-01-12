import numpy as np

class TrussSolver:

  @staticmethod
  def PlaneElementLength(x1, x2, y1, y2):
    y = np.sqrt([(x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)])
    if x1 == x2:
      theta = 90
    if x1 != x2:
      theta = np.arctan((y2-y1)/(x2-x1)) * 180 / np.pi
    if theta < 0:
      theta = 180 + theta
    return y, theta

  @staticmethod
  def PlaneElementStiffness(E, A, L, theta):
    x = theta * np.pi / 180
    C = np.cos(x)
    S = np.sin(x)
    Matrix = np.array([[C*C,C*S,-C*C,-C*S],[C*S,S*S,-C*S,-S*S],
                      [-C*C,-C*S,C*C,C*S],[-C*S,-S*S,C*S,S*S]])
    return (E * A / L) * Matrix

  @staticmethod
  def PlaneAssemble(K, k, i, j):
    K[2*i-2,2*i-2] = K[2*i-2,2*i-2] + k[0,0]
    K[2*i-2,2*i-1] = K[2*i-2,2*i-1] + k[0,1]
    K[2*i-2,2*j-2] = K[2*i-2,2*j-2] + k[0,2]
    K[2*i-2,2*j-1] = K[2*i-2,2*j-1] + k[0,3]
    K[2*i-1,2*i-2] = K[2*i-1,2*i-2] + k[1,0]
    K[2*i-1,2*i-1] = K[2*i-1,2*i-1] + k[1,1]
    K[2*i-1,2*j-2] = K[2*i-1,2*j-2] + k[1,2]
    K[2*i-1,2*j-1] = K[2*i-1,2*j-1] + k[1,3]
    K[2*j-2,2*i-2] = K[2*j-2,2*i-2] + k[2,0]
    K[2*j-2,2*i-1] = K[2*j-2,2*i-1] + k[2,1]
    K[2*j-2,2*j-2] = K[2*j-2,2*j-2] + k[2,2]
    K[2*j-2,2*j-1] = K[2*j-2,2*j-1] + k[2,3]
    K[2*j-1,2*i-2] = K[2*j-1,2*i-2] + k[3,0]
    K[2*j-1,2*i-1] = K[2*j-1,2*i-1] + k[3,1]
    K[2*j-1,2*j-2] = K[2*j-1,2*j-2] + k[3,2]
    K[2*j-1,2*j-1] = K[2*j-1,2*j-1] + k[3,3]
    return K

  @staticmethod
  def PlaneElementForce(E, A, L, theta, u):
    x = theta * np.pi / 180
    C = np.cos(x)
    S = np.sin(x)
    Matrix = np.array([-C,-S,C,S]).T
    return (E * A / L) * Matrix @ u

  @staticmethod
  def PlaneElementStress(E, L, theta, u):
    x = theta * np.pi / 180
    C = np.cos(x)
    S = np.sin(x)
    Matrix = np.array([-C,-S,C,S]).T
    return (E / L) * Matrix @ u

  @staticmethod
  def PlaneInclinedSupport(T, i, alpha):
    x = alpha * np.pi /180
    T[2*i-2,2*i-2] = np.cos(x)
    T[2*i-2,2*i-1] = np.sin(x)
    T[2*i-1,2*i-2] = -np.sin(x)
    T[2*i-1,2*i-1] = np.cos(x)
    return T