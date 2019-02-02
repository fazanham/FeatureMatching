"""
Created on Sat January 4 14:00:41 2017
A script for determining inliers using RANSAC
@author: Faizaan Naveed
"""

import numpy as np
import Transformations

def SumEucDistance(P1, P1_Co, P2, P2_Co):
    # Computes the sum of the euclidean distance between P1 and P1_Co and P2 and P2_Co
    # P1_Co represents the transformed coordinates
    # P2_Co represents the inverse transformed coordinates

    d1 = np.sqrt(np.power(P1[:,0]-P1_Co[:,0], 2) + np.power(P1[:,1]-P1_Co[:,1], 2))
    d2 = np.sqrt(np.power(P2[:,0]-P2_Co[:,0], 2) + np.power(P2[:,1]-P2_Co[:,1], 2))

    return d1+d2

def Ransac(Matches, TYPE, N, Epsilon):
    # Random sample consensus(RANSAC) is an iterative method to estimate parameters
    # of a mathematical model from a set of observed data that contains outliers.
    # By iteratively fitting a model to the data we find the best model by its ability
    # to characterize a threshold of inlier points.

    # Input: Corresponding Image coordinates in the two images (Matches)
    #        Threshold number of inlier points (N)
    #        Threshold value for Sum of Squared differences (Epsilon)
    #        TYPE determines the model to compute (1=Affine, 2=Homography)
    # Outputs: Number of inlier points
    #          Affine or Homography transformation matrix

    # Inlier count
    In_count = 0

    # Inlier array
    Inlier_Data = []

    while (In_count < N):
        # Randomly pick out points from the matches
        r = np.round((len(Matches) - 1)*np.random.random(np.int(np.round(len(Matches)/2)) + 1)).astype(int)
        Subsample = Matches[r,:]

        # Keep track of the inliers
        if (In_count > 0):
            Subsample = np.concatenate((Subsample, np.asarray(Inlier_Data)), axis=0)

        # Isolate the matches
        Image_Coords_1 = np.asarray(Subsample)[:,0:2]
        Image_Coords_2 = np.asarray(Subsample)[:,2:4]

        # Homography
        if (TYPE == 2):

            # Compute the homography matrix
            H = Transformations.Homography(Image_Coords_1, Image_Coords_2)

            # Compute the transformation and the inverse transformation
            X2 = Transformations.HomographyTransformer(H, Matches[:, 0:2])
            X1 = Transformations.HomographyTransformer(np.linalg.inv(H), Matches[:, 2:4])

            # Compute the euclidean distance
            ssd = SumEucDistance(X1, Matches[:, 0:2], X2, Matches[:, 2:4])

        # Affine
        elif (TYPE == 1):

            # Compute the affine matrix
            A = Transformations.AffineTransformer(Image_Coords_1, Image_Coords_2)

            # Compute the transformation coordinates and inverse transformation coordinates
            X2 = np.matmul(A, np.transpose(np.hstack((np.asarray(Matches[:,0:2]), np.ones(shape=(len(Matches),1))))))
            X1 = np.matmul(np.linalg.inv(A), np.transpose(np.hstack((np.asarray(Matches[:,2:4]), np.ones(shape=(len(Matches),1))))))

            # Compute the euclidean distance
            ssd = SumEucDistance(X1[0:2].T, Matches[:, 0:2], X2[0:2].T, Matches[:, 2:4])

        else:
            raise ValueError("Incorrect Type")

        # Extract the outliers
        for i in range(0, len(ssd)):
            if (ssd[i] < Epsilon):
                if [Matches[i,0],Matches[i,1],Matches[i,2],Matches[i,3]] not in np.asarray(Inlier_Data).tolist():
                    Inlier_Data.append(Matches[i,:])
                    In_count += 1
                    print(In_count)

    # Compute the transformation using the inliers
    Inlier_Data = np.asarray(Inlier_Data, dtype=float)

    if (TYPE == 1):
        return Inlier_Data, Transformations.AffineTransformer(Inlier_Data[:, 0:2], Inlier_Data[:, 2:4])
    elif (TYPE == 2):
        return Inlier_Data, Transformations.Homography(Inlier_Data[:, 0:2], Inlier_Data[:, 2:4])