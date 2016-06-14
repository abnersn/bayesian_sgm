#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import bayesian
import normalize

seg = bayesian.BayesianColorSGM()
seg.learn_from_dirs("datasets/classified", "datasets/unclassified")

cap = cv2.VideoCapture(0)
while(True):
	ret, img = cap.read()
	bin = seg.apply(img)
	cv2.imshow("frame",img)
	cv2.imshow("binary",bin)
	k = cv2.waitKey(30) & 0xFF
	if k == 27:
		break