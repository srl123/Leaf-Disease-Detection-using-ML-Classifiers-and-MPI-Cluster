# Leaf-Disease-Detection-using-ML-Classifiers-and-MPI-Cluster

# Abstract

Farmers cultivate a lot of crops all around the year. There is a lot of effort going into this process of agriculture. But these efforts effectively get hampered when these crops are inflicted with diseases. Many factors influence disease development in plants including hybrid/variety genetics, age of the plant at the time of infection, environment, weather, genetics of the pathogen populations etc. Those crops which are affected in turn may affect other crops also which are uninfluenced at the time of initial attack. So this poses a major threat to the effort of farmers especially those who have their only way of income/livelihood based on farming. So to find an effective solution to such a major problem and curb it effectively I developed a Leaf Disease Detection system using image processing techniques and tried to improve its performance using a MPI Cluster by using 2 Virtual Machines. In this project a performance analysis is also done to know about how much the speedup takes place when the system is run on a single node (1 VM) and on a 2-node cluster (2 VMs). I have also analyzed the accuracies using different Machine Learning Classifiers

# Objectives:

•	To detect leaf disease portion from image

•	To extract features of detected portion of leaf

•	To recognize detected portion of leaf through MPI Cluster and ML Classifiers

•	To compare and analyze the performance of detection on a single Virtual Machine (1 node) and 2 Virtual Machines (2 nodes)  and accuracies of ML Classifiers


# Applying Machine Learning

# 1.Image Segmentation
First, we take a set of 10 training images. With the images we have we are trying to segment the part which has disease present in it. For this purpose, we first convert the image to HSV format. There we take a range of HSV values (which was obtained after histogram analysis) b/w which are generally the pixel values for diseased part. Then we do bitwise and with the original image to get RGB version of diseased part

# 2.Feature Extraction
We then convert the image obtained from step 1 into Grayscale image. From that image we obtain the gray level co-occurrence matrix from wherein we get features like contrast, energy etc.
My project uses the following features:
Mean, Standard deviation, Variance, Energy, Contrast, Smoothness, Homogeneity, Entropy and RMS value

# 3.Training the Machine Learning Model 
We already have the set of features for all 10 images obtained now. Along with that the data of whether the leaf is healthy / unhealthy was obtained already. Using the features and the result obtained we train a machine learning model which helps us to figure out for a given image of leaf whether the leaf is healthy/unhealthy using Logistic Regression model. We used it because it gave very high results while execution for both training and test data.



# Parallelizing using 2 Virtual Machines


# 1.Splitting of test image and assigning to slave processes 
We split the test image of leaf on which we are going to apply the model to obtain the required results into 4. We obtain the center point of the image from which we get the required images. We then assign these to individual slave processes each of the images

# 2.Application of ML model on each thread’s image after feature extraction
We extract features which we discussed earlier in the previous sections from each process’s assigned image based on the same procedure. Then we apply the ML model to get the result status of that image which is assigned to that particular image. We then send the result status to the main master process.

# 3.Reduction to obtain the final result
Now based on the image status obtained from slave processes we assign whether the total leaf is unhealthy / healthy. The reduction process is based on the fact that even if one part is unhealthy the whole leaf should be detected unhealthy.
