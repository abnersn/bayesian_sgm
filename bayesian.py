#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys

class BayesianColorSGM(object):

	def __init__(self):
		self.post_h = None
		self.post_s = None
		self.post_v = None
		self.trained = False

	def learn_from_dirs(self, c_dir, n_dir):
		# Verifica se os diretórios existem
		for dir in [c_dir, n_dir]:
			if not os.path.exists(dir):
				print "ERROR: Could not load directory "+dir
				sys.exit(1)

		# Dados não classificados para cada canal serão acumulados em seus respectivos vetores
		h_data = np.empty([0],np.uint8())
		s_data = np.empty([0],np.uint8())
		v_data = np.empty([0],np.uint8())
		for f in os.listdir(n_dir):
			if f.endswith("jpg"):
				img = cv2.imread(n_dir+"/"+f)
				hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
				h, s, v = cv2.split(hsv)
				h_data = np.append(h_data, h.flatten())
				s_data = np.append(s_data, s.flatten())
				v_data = np.append(v_data, v.flatten())

		if len(h_data) == 0 or len(s_data) == 0 or len(v_data) == 0:
			print "ERROR: Could not load data from images"
			sys.exit(1)

		# Calcula e trata as probabilidades dos valores para cada canal
		prob_h = cv2.calcHist([h_data],[0],None,[180],[0,180]) / float(h_data.size)
		prob_s = cv2.calcHist([s_data],[0],None,[256],[0,256]) / float(s_data.size)
		prob_v = cv2.calcHist([v_data],[0],None,[256],[0,256]) / float(v_data.size)
		prob_h[prob_h == 0] = 1
		prob_s[prob_s == 0] = 1
		prob_v[prob_v == 0] = 1

		# Os dados em *_class são extraídos das amostras classificadas
		h_class = np.empty([0],np.uint8())
		s_class = np.empty([0],np.uint8())
		v_class = np.empty([0],np.uint8())

		for f in os.listdir(c_dir):
			if f.endswith("jpg"):
				img = cv2.imread(n_dir+"/"+f)
				msk = cv2.imread(c_dir+"/"+f,0)
				ret, bin = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
				hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				h, s, v = cv2.split(hsv)
				# Seleciona apenas os pixels marcados como pele
				h_class = np.append(h_class, h[bin > 127].flatten())
				s_class = np.append(s_class, s[bin > 127].flatten())
				v_class = np.append(v_class, v[bin > 127].flatten())

		# Calcula as probabilidades condicionais dos valores para cada canal
		prob_cond_h = cv2.calcHist([h_class],[0],None,[180],[0,180]) / float(h_class.size)
		prob_cond_s = cv2.calcHist([s_class],[0],None,[256],[0,256]) / float(s_class.size)
		prob_cond_v = cv2.calcHist([v_class],[0],None,[256],[0,256]) / float(v_class.size)

		# Calcula a porcentagem de pixels marcados como pele
		prob_class = h_class.size / float(h_data.size)

		# Calcula as probabilidades a posteriori para cada canal
		self.post_h = np.divide(prob_cond_h, prob_h) * prob_class
		self.post_s = np.divide(prob_cond_s, prob_s) * prob_class
		self.post_v = np.divide(prob_cond_v, prob_v) * prob_class
		
		self.trained = True
	
	def apply(self, img):
		if(not self.trained):
			print "ERROR: Use learn_from_dirs(c_dir, n_dir) before applying."
			sys.exit(1)
		hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(hsv)

		# Calcula as probabilidades para as coordenadas hue e saturation da imagem
		calc = np.multiply(self.post_h[h],self.post_s[s]) * 255
		ret, bin = cv2.threshold(calc, 10, 255, cv2.THRESH_BINARY)
		return bin
