#cs16 review chapter 1

'''
image resizing, delete unimportant pixels, look at neighbors to differentiate
between important colors, calc as sum difference between pixels for importance
value

data = [[0] * cols for i in range(rows)]
numpy.argmin(list) --> index of min of list or function
min --> value of min of list or function, or arguments

remove seams from photo, vertical or horizontal
'''
import random
from pprint import pprint
from copy import deepcopy
import numpy 

picture = [[0]*20 for i in range(20)]
for i in range(20):
	for j in range(20):
		picture[i][j] = random.randrange(20)

def attempt(picture):
	importance = calculateImportance(picture)
	seam = findSeam(importance)

def calculateImportance(picture):
	row = len(picture)
	col = len(picture[0])
	importance = [[0]*col for i in range(row)]
	for i in range(0,row):
		for j in range(0,col):
			if i == 0 and j == 0:
				importance[i][j] = (abs(picture[i][j]-picture[i+1][j]) + abs(picture[i][j]-picture[i][j+1]))/2
			elif i == row-1 and j == col-1:
				importance[i][j] = (abs(picture[i][j]-picture[i-1][j]) + abs(picture[i][j]-picture[i][j-1]))/2	
			elif i == 0 and j == col-1:
				importance[i][j] = (abs(picture[i][j]-picture[i+1][j]) + abs(picture[i][j]-picture[i][j-1]))/2
			elif i == row-1 and j == 0:
				importance[i][j] = (abs(picture[i][j]-picture[i-1][j]) + abs(picture[i][j]-picture[i][j+1]))/2
			elif i == 0:
				importance[i][j] = (abs(picture[i][j]-picture[i][j-1]) + abs(picture[i][j]-picture[i][j+1]) + abs(picture[i][j]-picture[i+1][j]))/3
			elif j == 0:
				importance[i][j] = (abs(picture[i][j]-picture[i-1][j]) + abs(picture[i][j]-picture[i+1][j]) + abs(picture[i][j]-picture[i][j+1]))/3
			elif i == row-1:
				importance[i][j] = (abs(picture[i][j]-picture[i][j-1]) + abs(picture[i][j]-picture[i][j+1]) + abs(picture[i][j]-picture[i-1][j]))/3
			elif j == col-1:
				importance[i][j] = (abs(picture[i][j]-picture[i-1][j]) + abs(picture[i][j]-picture[i+1][j]) + abs(picture[i][j]-picture[i][j-1]))/3
			else:
				importance[i][j] = (abs(picture[i][j]-picture[i-1][j]) + abs(picture[i][j]-picture[i+1][j]) + abs(picture[i][j]-picture[i][j-1]) + abs(picture[i][j]-picture[i][j+1]))/4
	return importance

def findSeam(importance):
	seams = deepcopy(importance)
	col = len(importance)
	row = len(importance[0])
	directions = [[0]*col for i in range(row)]
	for i in range(1,row):
		for j in range(0,col):
			if j == 0 and j == col-1:
				seams[i][j] += seams[i-1][j]
				directions[i][j] = 0				
			elif j == 0:
				if seams[i-1][j] <= seams[i-1][j+1]:
					seams[i][j] += seams[i-1][j]	
					directions[i][j] = 0
				else:
					seams[i][j] += seams[i-1][j+1]	
					directions[i][j] = 1
			elif j == col-1:
				if seams[i-1][j-1] <= seams[i-1][j]:
					seams[i][j] += seams[i-1][j-1]	
					directions[i][j] = -1
				else:
					seams[i][j] += seams[i-1][j]	
					directions[i][j] = 0
			else:
				if seams[i-1][j-1] <= seams[i-1][j] and seams[i-1][j-1] <= seams[i-1][j+1]:
					seams[i][j] += seams[i-1][j-1]
					directions[i][j] = -1
				elif seams[i-1][j] <= seams[i-1][j+1]:
					seams[i][j] += seams[i-1][j]	
					directions[i][j] = 0
				else:
					seams[i][j] += seams[i-1][j+1]	
					directions[i][j] = 1
	start = numpy.argmin(seams[-1])
	seam = [[0] for i in range(row)]
	k = start
	n = row-1
	for i in range(row):
		seam[i] = directions[n][k]
		if seams[i] == -1:
			k -= 1
		elif seams[i] == 1:
			k += 1
		n -= 1
	return seam

if __name__ == '__main__':
	attempt(picture)
