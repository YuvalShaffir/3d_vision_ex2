import numpy as np
import matplotlib.pyplot as plt
import cv2
import src.sfm as sfm
import src.homography as hg

# load the images
img1 = cv2.imread('E:\\PythonProjects\\3D_vision_ex2\\matchbox1.jpg')
img2 = cv2.imread('E:\\PythonProjects\\3D_vision_ex2\\matchbox2.jpg')

# convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# create the SIFT detector
sift = cv2.SIFT_create()

# detect keypoints and descriptors
kp1, dp1 = sift.detectAndCompute(gray1, None)
img3 = cv2.drawKeypoints(gray1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('image', img3)
cv2.waitKey(0)

kp2, dp2 = sift.detectAndCompute(gray2, None)
img4 = cv2.drawKeypoints(gray2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('image', img4)
cv2.waitKey(0)

# match points
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(dp1, dp2)
# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)
# Extract matched points
x1 = np.zeros((len(matches), 2), dtype=np.float32)
x2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    x1[i, :] = kp1[match.queryIdx].pt
    x2[i, :] = kp2[match.trainIdx].pt

x1 = hg.make_homogeneous(x1.T)
x2 = hg.make_homogeneous(x2.T)
K = np.array([[2394, 0, 932], [0, 2398, 628], [0, 0, 1]])
x1n = np.dot(np.linalg.inv(K), x1)
x2n = np.dot(np.linalg.inv(K), x2)

# find the fundamental matrix
model = sfm.RansacModel()
# F, inliers = sfm.F_from_ransac(x1n, x2n, model)
F, inliers = cv2.findFundamentalMat(x1.T, x2.T, cv2.FM_RANSAC)
print("The fundamental matrix: ", F)

e1, e2 = sfm.compute_epopole(F)
print(f"epipole 1: {e1} \n epipole 2: {e2}")
lines1 = []
lines2 = []
fig = plt.figure()
plt.imshow(img2)
for i in range(30):
    lines2.append(sfm.plot_epipolar_line(img2, F, x2[:, i], e2, True))
plt.axis('off')

figure = plt.figure()
plt.imshow(img1)
for i in range(30):
    lines1.append(sfm.plot_epipolar_line(img1, F.T, x1[:, i], e1, True))
plt.axis('off')

plt.show()

def find_matches_along_epipolar_lines(points1, points2, lines1, lines2, threshold=1.0):
    matches = []
    for i, (pt1, l2) in enumerate(zip(points1, lines2)):
        for j, (pt2, l1) in enumerate(zip(points2, lines1)):
            distance1 = abs(l2[0] * pt1[0] + l2[1] * pt1[1] + l2[2]) / np.sqrt(l2[0] ** 2 + l2[1] ** 2)
            distance2 = abs(l1[0] * pt2[0] + l1[1] * pt2[1] + l1[2]) / np.sqrt(l1[0] ** 2 + l1[1] ** 2)
            if distance1 < threshold and distance2 < threshold:
                matches.append((i, j))
    return matches


refined_matches = find_matches_along_epipolar_lines(x1.T, x2.T, lines1, lines2)
print(f"more matches {refined_matches}")

E = K.T @ F @ K
# compute camera matrices
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = sfm.compute_P_from_essential(E)
print("The camera matrices: ", P1, P2)

X = sfm.triangulate(x1n, x2n, P1, P2[0])
# # pick the solution with points in front of cameras
# i = 0
# max_res = 0
# infront = None
# # Loop over possible camera poses
# for j in range(2):
#     # Triangulate inliers and compute depth for each camera
#     X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[j])
#     d1 = np.dot(P1, X)[2]
#     d2 = np.dot(P2[j], X)[2]
#
#     # Count the number of points in front of both cameras
#     if np.sum(d1 > 0) + np.sum(d2 > 0) > max_res:
#         max_res = np.sum(d1 > 0) + np.sum(d2 > 0)
#         i = j
#         infront = (d1 > 0) & (d2 > 0)
#
# # Triangulate inliers with the best camera pose
# X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
# X = X[:, infront]  # keep only points infront of both cameras

# 3D plot

# First plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Using add_subplot to explicitly set 3D projection
ax.scatter(-X[0], X[1], X[2], c='k', marker='.')
ax.set_title("Original 3D Points")
ax.axis('on')

# plot the projection of X
import camera
# project 3D points
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[0])
x1p = cam1.project(X)
x2p = cam2.project(X)
# reverse K normalization
x1p = np.dot(K, x1p)
x2p = np.dot(K, x2p)
plt.figure()
plt.imshow(img1)
# gray()
plt.plot(x1p[0], x1p[1],'o')
plt.plot(x1[0], x1[1],'r.')
plt.axis('off')
plt.figure()
plt.imshow(img2)
# gray()
plt.plot(x2p[0], x2p[1],'o')
plt.plot(x2[0], x2[1],'r.')
plt.axis('off')
plt.show()

# Second plot
fig1 = plt.figure()
x1_3d = np.dot(P1, X)
for i in range(3):
    x1_3d[i] /= x1_3d[2]
x1_3d = np.dot(K, x1_3d)
ax1 = fig1.add_subplot(111, projection='3d')  # Using add_subplot to explicitly set 3D projection
ax1.scatter(x1_3d[0], x1_3d[1], x1_3d[2], c='k', marker='.')
ax1.set_title("Transformed 3D Points with P1")
ax1.axis('on')

# Third plot
fig2 = plt.figure()
x2_3d = np.dot(P2[0], X)
for i in range(3):
    x2_3d[i] /= x2_3d[2]
x2_3d = np.dot(K, x2_3d)
ax2 = fig2.add_subplot(111, projection='3d')  # Using add_subplot to explicitly set 3D projection
ax2.scatter(x2_3d[0], x2_3d[1], x2_3d[2], c='k', marker='.')
ax2.set_title("Transformed 3D Points with P2")
ax2.axis('on')

# Show all plots
plt.show()