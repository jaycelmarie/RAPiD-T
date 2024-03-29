#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import math 
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity


def get_video_frame_iterator(filename, from_python_2=False):
	with open(filename, 'rb') as f:
		while True:
			try:
				if from_python_2:
					yield pickle.load(f, encoding='latin1')
				else:
					yield pickle.load(f)
			except EOFError:
				return
			except Exception as e:
				print('Unable to load data ', filename, ':', e)
				raise ValueError('Unable to load data ', filename, ':', e)
		
		

# Computes a set of similarity features between two detections
def get_pair_features(p1, p2, feat_names=[]): 			# , iw, ih ||| , euclidean_dist=True, correlation=True
	
	for p in [p1, p2]:
		if len(p["bbox"]) == 5:
			x0, y0, w, h, angle = p["bbox"]
			if w > h:
				xc = x0 + w/2; yc = y0 + h/2
				temp = w; w = h; h = temp; angle += 90
				x0 = xc - w/2; y0 = yc - h/2
			while angle > 90:
				angle -= 180
			while angle < -90:
				angle += 180
			p["bbox"] = [x0, y0, w, h, angle]

	feats = {}
	
	if 'width_rel' in feat_names or len(feat_names)==0:
		feats['width_rel'] = min(p1['bbox'][2], p2['bbox'][2]) / max(p1['bbox'][2], p2['bbox'][2])
	if 'height_rel' in feat_names or len(feat_names)==0:
		feats['height_rel'] = min(p1['bbox'][3], p2['bbox'][3]) / max(p1['bbox'][3], p2['bbox'][3])
	if 'angle_diff' in feat_names:
		assert -90 <= p1['bbox'][4] <= 90, print(p1['bbox'])
		assert -90 <= p2['bbox'][4] <= 90, print(p2['bbox'])
		angle_diff = abs(p1['bbox'][4] - p2['bbox'][4])
		angle_diff = min(angle_diff, 180 - angle_diff)/90
		assert 0 <= angle_diff <= 1, print(angle_diff)

		feats['angle_diff'] = angle_diff

	if 'iou' in feat_names or len(feat_names)==0:
		feats['iou'] = get_iou(p1['bbox'].copy(), p2['bbox'].copy())
	if 'iou_rotated' in feat_names or len(feat_names)==0:
		# feats['iou_rotated'] = get_iou(p1['bbox'].copy(), p2['bbox'].copy())
		feats['iou_rotated'] = get_iou_rotated(p1['bbox'].copy(), p2['bbox'].copy())

	if 'center_distances_corrected' in feat_names or len(feat_names)==0:
		feats['center_distances_corrected'] = euclidean_distance_between_centers_corrected(p1['bbox_center'], p2['bbox_center'])

	if 'descriptor_dist' in feat_names or len(feat_names)==0:
		feats['descriptor_dist'] = np.linalg.norm(p1['emb'] - p2['emb'])

	if 'descriptor_cos' in feat_names or len(feat_names)==0:
		feats['descriptor_cos'] = np.dot(p1['emb'].reshape(-1), p2['emb'].reshape(-1)) \
								  / (np.linalg.norm(p1['emb'].reshape(-1)) * np.linalg.norm(p2['emb'].reshape(-1)))
		
	return feats


def euclidean_distance_between_centers_corrected(center1, center2):
	return math.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)



def get_iou(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x1, y1) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner
	bb2 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x, y) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	# Convert width/height to coordinates
	bb1[2] += bb1[0]; bb1[3] += bb1[1]
	bb2[2] += bb2[0]; bb2[3] += bb2[1]
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou


def get_iou_rotated(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two rotated bounding boxes.

	Parameters
	----------
	bb1, bb2 	: dict
				Keys: {'x', 'y', 'w', 'h', 'angle'}
				The (x, y) position is at the top left corner,
				the (w, h, angle) are width, height and angle

	Returns
	-------
	float
		in [0, 1]
	"""
	# Convert width/height to coordinates
	bb1[2] += bb1[0]; bb1[3] += bb1[1]
	bb2[2] += bb2[0]; bb2[3] += bb2[1]
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

	a = Polygon([(bb1[0], bb1[1]), (bb1[2], bb1[1]), (bb1[2], bb1[3]), (bb1[0], bb1[3])])
	b = Polygon([(bb2[0], bb2[1]), (bb2[2], bb2[1]), (bb2[2], bb2[3]), (bb2[0], bb2[3])])
	a = affinity.rotate(a, bb1[4], 'center')
	b = affinity.rotate(b, bb2[4], 'center')
	iou = a.intersection(b).area / a.union(b).area

	assert iou >= 0.0
	assert iou <= 1.0
	return iou

