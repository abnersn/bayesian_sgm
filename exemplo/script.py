import cv2
import numpy as np
import bayesian_sgm

seg = bayesian_sgm.BayesianColorSGM()
seg.learn_from_dirs("datasets/c_dataset", "datasets/n_dataset")

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    bin = seg.apply(img)
    cv2.imshow("frame", img)
    cv2.imshow("binary", bin)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
