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
F, inliers = sfm.F_from_ransac(x1n, x2n, model)
print("The fundamental matrix: ", F)

e = sfm.compute_epopole(F)
fig = plt.figure()
plt.imshow(img1)
for i in range(30):
    sfm.plot_epipolar_line(img1, F, x2n[:, i], e, False)
plt.axis('off')

figure = plt.figure()
plt.imshow(img2)
for i in range(30):
    sfm.plot_epipolar_line(img2, F.T, x1n[:, i], e, False)
plt.axis('off')

plt.show()


E = K.T @ F @ K
# compute camera matrices
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_essential(E)
print("The camera matrices: ", P1, P2)

# pick the solution with points in front of cameras
i = 0
max_res = 0
infront = None
# Loop over possible camera poses
for j in range(2):
    # Triangulate inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[j])
    d1 = np.dot(P1, X)[2]
    d2 = np.dot(P2[j], X)[2]

    # Count the number of points in front of both cameras
    if np.sum(d1 > 0) + np.sum(d2 > 0) > max_res:
        max_res = np.sum(d1 > 0) + np.sum(d2 > 0)
        i = j
        infront = (d1 > 0) & (d2 > 0)

# Triangulate inliers with the best camera pose
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
X = X[:, infront]  # keep only points infront of both cameras

# plot the 3D points
fig = plt.figure()

# Create a 3D axis
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.plot(-X[0], X[1], X[2], 'k.')

ax.set_axis_off()
plt.show()
