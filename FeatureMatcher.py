"""
Created on Sat January 4 14:00:41 2017
Feature matcher main class
@author: Faizaan Naveed
"""

import Ransac
import Transformations
import numpy as np
import matplotlib.pyplot as plt

I1 = plt.imread('library1.jpg')
I2 = plt.imread('library2.jpg')

matches = np.loadtxt('library_matches.txt')

Input_coord1 = matches[:,0:2]
Input_coord2 = matches[:,2:4]

### LS Affine transformation ###
Affine_T = Transformations.AffineTransformer(Input_coord1, Input_coord2)

# Generate a pseudo z-axis
PS_Input_coord1 = np.hstack((Input_coord1, np.ones(shape=(len(Input_coord1), 1))))
PS_Input_coord2 = np.hstack((Input_coord2, np.ones(shape=(len(Input_coord2), 1))))

# Transformed coordinates
Tr_coord1 = np.matmul(np.linalg.inv(Affine_T),np.transpose(PS_Input_coord2))
Tr_coord2 = np.matmul(Affine_T,np.transpose(PS_Input_coord1))

# Residuals and mean error
Residuals_IC1 = Tr_coord2.T-PS_Input_coord2
Residuals_IC2 = Tr_coord1.T-PS_Input_coord1

Mean_Res_IC1 = np.mean(Residuals_IC1)
Mean_Res_IC2 = np.mean(Residuals_IC2)

# Display the image
fig = plt.figure('LS Affine Transformation')
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(I1)
plt.axis('off')
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(I2)
plt.axis('off')

# Plot the transformed points
ax2.plot(matches[:,2], matches[:,3], 'r+')
ax2.plot(Tr_coord2[0,:], Tr_coord2[1,:], 'b+')

ax1.plot(matches[:,0], matches[:,1], 'r+')
ax1.plot(Tr_coord1[0,:], Tr_coord1[1,:], 'b+')

### Affine + Ransac ###
Inliers, Affine_R = Ransac.Ransac(matches, TYPE=1, N=10, Epsilon=0.1)

Tr_coord1_R = np.matmul(np.linalg.inv(Affine_R),np.transpose(PS_Input_coord2))
Tr_coord2_R = np.matmul(Affine_R,np.transpose(PS_Input_coord1))

# Display the image
fig = plt.figure('LS Affine Ransac')

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(I1)
plt.axis('off')
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(I2)
plt.axis('off')

# Plot the transformed points
ax2.plot(matches[:,2], matches[:,3], 'r+')
ax2.plot(Tr_coord2_R[0,:], Tr_coord2_R[1,:], 'b+')

ax1.plot(matches[:,0], matches[:,1], 'r+')
ax1.plot(Tr_coord1_R[0,:], Tr_coord1_R[1,:], 'b+')

### Homography Transformation ###
Homography_T = Transformations.Homography(Input_coord1, Input_coord2)

# Transformed coordinates
Tr_coord2_H = Transformations.HomographyTransformer(Homography_T, Input_coord1)
Tr_coord1_H = Transformations.HomographyTransformer(np.linalg.inv(Homography_T), Input_coord2)

# Display the image
fig = plt.figure('Homogrphy Transformation')
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(I1)
plt.axis('off')
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(I2)
plt.axis('off')

# Plot the transformed points
ax2.plot(matches[:,2], matches[:,3], 'r+')
ax2.plot(Tr_coord2_H[:,0], Tr_coord2_H[:,1], 'b+')

ax1.plot(matches[:,0], matches[:,1], 'r+')
ax1.plot(Tr_coord1_H[:,0], Tr_coord1_H[:,1], 'b+')

### Homography Ransac Transformation ###
Inliers, Homography_R = Ransac.Ransac(matches, TYPE=2, N=10, Epsilon=0.1)

# Transformed coordinates
Tr_coord2_H = Transformations.HomographyTransformer(Homography_R, Input_coord1)
Tr_coord1_H = Transformations.HomographyTransformer(np.linalg.inv(Homography_R), Input_coord2)

# Display the image
fig = plt.figure('Homogrphy Ransac Transformation')
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(I1)
plt.axis('off')
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(I2)
plt.axis('off')

# Plot the transformed points
ax2.plot(matches[:,2], matches[:,3], 'r+')
ax2.plot(Tr_coord2_H[:,0], Tr_coord2_H[:,1], 'b+')

ax1.plot(matches[:,0], matches[:,1], 'r+')
ax1.plot(Tr_coord1_H[:,0], Tr_coord1_H[:,1], 'b+')

print('All done!')