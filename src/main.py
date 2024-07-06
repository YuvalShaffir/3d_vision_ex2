import numpy as np
import matplotlib.pyplot as plt
import cv2
import src.sfm as sfm

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
matches = sorted(matches, key=lambda x: x.distance)

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# find the fundamental matrix
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
print("The fundamental matrix: ", F)


# find the epipole
U, S, Vt = np.linalg.svd(F)
e1 = Vt[-1]
e2 = U[:, -1]

e1 = e1 / e1[2]  # Normalize
e2 = e2 / e2[2]  # Normalize

print(f'Epipole in image 1: {e1}')
print(f'Epipole in image 2: {e2}')

lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)


def draw_lines(img, lines, points):
    height, width = img.shape
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line, pt in zip(lines, points):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [width, -(line[2] + line[0] * width) / line[1]])
        img_color = cv2.line(img_color, (x0, y0), (x1, y1), color, 1)
        img_color = cv2.circle(img_color, tuple(int(x) for x in pt), 5, color, -1)
    return img_color


img1_lines = draw_lines(gray1, lines1, points1)
img2_lines = draw_lines(gray2, lines2, points2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img1_lines, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_lines, cmap='gray')
plt.title('Image 2')
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


refined_matches = find_matches_along_epipolar_lines(points1, points2, lines1, lines2)

# Reconstruct 3D Points:

# Camera matrices (for this example, assume identity matrices)
# P1 = np.eye(3, 4)
K = np.array([[2394, 0, 932], [0, 2398, 628], [0, 0, 1]])

E = K.T @ F @ K
# compute camera matrices
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
# Decompose the Essential Matrix E using SVD
U, _, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# Possible rotations and translations
R1 = np.matmul(U, np.matmul(W, Vt))
R2 = np.matmul(U, np.matmul(W.T, Vt))
t1 = U[:, 2]
t2 = -U[:, 2]

# Construct possible camera matrices P2
P2_1 = np.hstack((R1, t1.reshape(3, 1)))
P2_2 = np.hstack((R1, t2.reshape(3, 1)))
P2_3 = np.hstack((R2, t1.reshape(3, 1)))
P2_4 = np.hstack((R2, t2.reshape(3, 1)))

# Choose the correct P2 based on the determinant condition
P2_options = [P2_1, P2_2, P2_3, P2_4]
for P2 in P2_options:
    if np.linalg.det(P2[:, :3]) > 0:
        break

# Optionally, you can convert points to homogeneous coordinates for triangulation
points1_h = np.vstack((points1.T, np.ones((1, points1.shape[0]))))
points2_h = np.vstack((points2.T, np.ones((1, points2.shape[0]))))

# Perform triangulation
points_3d_h = cv2.triangulatePoints(P1, P2, points1_h[:2], points2_h[:2])

# Convert from homogeneous coordinates to 3D
points_3d = points_3d_h[:3] / points_3d_h[3]

# Transpose to have the correct shape
points_3d = points_3d.T

# Visualize the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
plt.show()
