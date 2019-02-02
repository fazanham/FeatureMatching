# PythonFeatureMatching
This is a software utility for feature matching using affine and homography transformations. The software can be used in cohesion with feature detectors (i.e. SIFT and FLANN) to compute the transformations. To mitigate the effect of outliers, I also provide an implementation of RANSAC.

# Dependencies
Numpy<br/>
Matplotlib<br/>

# Transformations
The script provides two transformations: Affine transformation (a linear transformation that preserves the parallelism of lines) and Homography transformation (a projective transformation).<br/><br/>
Since the Affine matrix has 6 DoFs, you need a minimum of 3 corresponding pairs to solve the problem.<br/>
For Homography matrix there are 8 DoFs, so you need a minimum of 4 corresponding pairs.<br/><br/>
I have also provided a function for shifting the coordinates to a centroid system, as in the case of large coordinate values, the matrices can become instable during inversion.

# RANSAC
This script provides an implementation of RANSAC for computing the transformations using a selected number of inlier points. The Euclidean distance metric is used to determine the 'outlingness' of the points. You have to provide the minimum number of inliers you want to detect and a threshold for classifying a point pair as an inlier. 

# Example
Affine:<br/>
![house_affine](https://user-images.githubusercontent.com/33495209/52169837-8a022d00-270c-11e9-9289-37ad97689b0c.PNG)

Affine + RANSAC:
![house_affine_ransac](https://user-images.githubusercontent.com/33495209/52169840-98e8df80-270c-11e9-9c4f-8ebcf3a23df2.PNG)

Homography:<br/>
![homography_library](https://user-images.githubusercontent.com/33495209/52169842-a3a37480-270c-11e9-9ad6-bd1092ab4555.PNG)

Homography + RANSAC:
![homography_ransac](https://user-images.githubusercontent.com/33495209/52169845-ad2cdc80-270c-11e9-8baf-76ee268295a1.PNG)

## License
Free-to-use (MIT), but at your own risk.
