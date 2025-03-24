import os
import numpy as np
from PIL import Image as im

from eigenfunction import eigen

def train_images(path):
	faces_matrix = []
	pict_name = []

	picts = os.listdir(path)
	for pict in picts:
		print("Processing: " + pict)
		img = im.open(path + "/" + pict).convert("L")
		img = img.resize((256,256), im.BICUBIC)
		img = np.array(img).flatten()
		faces_matrix.append(img)
		pict_name.append(path + "/" + pict)

	pict_name = np.array(pict_name)

	# m adalah jumlah wajah dalam dataset
	m = len(faces_matrix)

	# faces_matrix -> M x N^2
	faces_matrix = np.array(faces_matrix)

	# hitung mean face
	mean_face = np.mean(faces_matrix, axis=0)

	faces_norm = []
	# "normal"kan wajah training (face - average face)
	for i in range(m):
		faces_norm.append(faces_matrix[i] - mean_face)

	faces_norm = np.array(faces_norm)
	# faces_norm adalah "normal" dari wajah
	faces_norm = np.transpose(faces_norm)
	# faces_matrix -> N^2 x M

	# C = A^T x A
	C = np.transpose(faces_norm) @ faces_norm
	# Matriks kovarian, C -> M x M

	print("Processing eigenvalues...")
	# Hitung nilai eigen dan vektor eigen C
	EigVal, EigVect = eigen(C, iteration=15000)
	# EigVect adalah matriks dengan entri kolom ke-i vektor basis yang berkoresponden dengan nilai eigen ke-i EigVal

	print("Processing eigenfaces...")
	# EigFace adalah matriks wajah-wajah eigen, dengan entri baris adalah vektor wajah
	EigFace = []
	for i in range(m):
		EigFace.append(faces_norm @ EigVect[:,i])
		EigFace[i] = EigFace[i]/(np.linalg.norm(EigFace))

	EigFace = np.array(EigFace)
	# EigFace -> M x N^2

	# for i in range(m):
	# 	for j in range(i,m):
	# 		print(np.dot(EigFace[i], EigFace[j]))

	# nyatakan wajah-wajah sebagai kombinasi linier dalam Om
	Om = []
	for i in range(m):
		Om.append([np.dot(faces_norm[:,i], u) for u in EigFace])	# menghasilkan konstanta kelipatan suatu vektor eigen sebagai komponennya

	Om = np.array(Om)

	print("Training complete")

	return pict_name, mean_face, EigFace, Om


def test_image(img, pict_name, mean_face, EigFace, Om):
	img = img.convert("L").resize((256,256), im.BICUBIC)
	img = np.array(img).flatten()
	
	NewFace = img - mean_face
	Omk = [np.dot(NewFace, u) for u in EigFace]
	Omk = np.array(Omk)

	m = len(EigFace)
	min = 99999999
	k = -1
	for i in range(m):
		v = Omk - Om[i]	# hitung vektor selisih
		epsilon = np.sqrt(np.dot(v,v))	# hitung euclidean distance
		if(epsilon < min):
			min = epsilon
			k = i

	return pict_name[k]