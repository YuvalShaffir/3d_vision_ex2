import numpy as np
import ransac
import matplotlib.pyplot as plt


def compute_fundamental_matrix(x1, x2):
    """
    F is a 3x3 rank 2 matrix that satisfies x'Fx = 0 for all corresponding points x and x'.
    """
    # validate input shape is 3xN
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build the set of linear equations that F must satisfy: x'.T@F@x = 0
    a = np.zeros((n, 9))
    for i in range(n):
        a[i] = [
            x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i],
            x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i],
            x2[0, i], x2[1, i], 1
        ]

    # compute linear least square solution
    u, s, v = np.linalg.svd(a)
    f = v[-1].reshape(3, 3)

    # constrain F to rank 2
    u, s, v = np.linalg.svd(f)
    s[2] = 0
    f = np.dot(u, np.dot(np.diag(s), v))

    return f


def compute_epopole(f):
    """
    Compute the epipole from a fundamental matrix f.
    The epipole is a point where all epipolar lines intersect.
    """
    # compute the right null space of F = U*S*V.T and the last column of V is the epipole
    # find the epipole
    U, S, Vt = np.linalg.svd(f)
    e1 = Vt[-1]
    e2 = U[:, -1]

    e1 = e1 / e1[2]  # Normalize
    e2 = e2 / e2[2]  # Normalize # normalize so e is homogeneous
    return e1, e2


def plot_epipolar_line(img, F, x, epipole=None, show_epipole=True):
    m, n = img.shape[:2]
    line = np.dot(F, x)

    t = np.linspace(0, n, 100)
    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt > 0) & (lt < m)
    plt.plot(t[ndx], lt[ndx], linewidth=1)

    if show_epipole:
        if epipole is None:
            epipole = compute_epopole(F)
        # plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')
        epipole_x, epipole_y = epipole[0] / epipole[2], epipole[1] / epipole[2]

        # Plot the epipole if it is outside the image boundaries
        if epipole_x < 0 or epipole_x > n or epipole_y < 0 or epipole_y > m:
            plt.plot([epipole_x], [epipole_y], 'r*', markersize=5)
    return lt


def skew(a):
    """
    Skew matrix A such that a x v = Av for any v.
    :param a:
    :return:
    """
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def compute_P_from_fundamental(F):
    """
    Compute the second camera matrix P' from a fundamental matrix F.
    :param F:
    :return: P'
    """
    e = compute_epopole(F.T)  # left epipole
    e = e.reshape(3, 1)
    te = skew(e)
    return np.vstack((np.dot(te, F.T).T, e.T)).T


def compute_P_from_essential(E):
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V)) < 0:
        V = -V

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    P2 = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2


def triangulate(x1, x2, P1, P2):
    """
    Triangulate the 3D point X given two points x1 and x2 and the camera matrices P1 and P2.
    :param x1:
    :param x2:
    :param P1:
    :param P2:
    :return:
    """
    assert x1.shape[1] == x2.shape[1], "Number of points must be the same"

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
    return np.array(X).T


def triangulate_point(x1, x2, P1, P2):
    M = np.zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2

    _, _, V = np.linalg.svd(M)
    X = V[-1, :4]

    # num_points = x1.shape[1]
    # X = np.zeros((4, num_points))  # Homogeneous coordinates
    #
    # for i in range(num_points):
    #     # Construct matrix A for the i-th point
    #     A = np.zeros((4, 4))
    #     A[0] = x1[0, i] * P1[2] - P1[0]
    #     A[1] = x1[1, i] * P1[2] - P1[1]
    #     A[2] = x2[0, i] * P2[2] - P2[0]
    #     A[3] = x2[1, i] * P2[2] - P2[1]
    #
    #     # Solve for X by minimizing ||AX||
    #     _, _, V = np.linalg.svd(A)
    #     X[:, i] = V[-1]

    # Convert homogeneous coordinates to 3D
    X /= X[3]

    return X


class RansacModel(object):
    def __init__(self):
        pass

    def fit(self, data):
        # split data into the two point sets
        data = data.T
        x1 = data[:3, :]
        x2 = data[3:, :]

        return compute_fundamental_matrix(x1, x2)

    def get_error(self, data, F):
        """
        Compute the error of the fundamental matrix F given two sets of corresponding points x1 and x2.
        :param F:
        :return:
        """
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        Fx1 = np.dot(F, x1)
        Fx2 = np.dot(F, x2)
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (np.diag(np.dot(x2.T, np.dot(F, x1))) ** 2) / denom
        return err

    def compute_f_from_normalized_points(self, x1, x2):
        """
        Compute the fundamental matrix from normalized points.
        :param x1:
        :param x2:
        :return:
        """
        n = x1.shape[1]
        if x2.shape[1] != n:
            raise ValueError("Number of points don’t match.")

        # normalize the points
        x1 = x1 / x1[2]
        mean1 = np.mean(x1[:2], axis=1)
        S1 = np.sqrt(2) / np.std(x1[:2])
        T1 = np.array([[S1, 0, -S1 * mean1[0]], [0, S1, -S1 * mean1[1]], [0, 0, 1]])
        x1 = np.dot(T1, x1)

        x2 = x2 / x2[2]
        mean2 = np.mean(x2[:2], axis=1)
        S2 = np.sqrt(2) / np.std(x2[:2])
        T2 = np.array([[S2, 0, -S2 * mean2[0]], [0, S2, -S2 * mean2[1]], [0, 0, 1]])
        x2 = np.dot(T2, x2)

        # compute the fundamental matrix
        F = compute_fundamental_matrix(x1, x2)

        # denormalize the fundamental matrix
        F = np.dot(T2.T, np.dot(F, T1))

        return F


def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    data = np.vstack((x1, x2))
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_threshold, 20, return_all=True)
    return F, ransac_data['inliers']
