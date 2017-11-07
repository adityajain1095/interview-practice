from pprint import pprint
import string
#
# #unique char in string
# def one_one(string):
# 	word_dic = set()
# 	for char in string:
# 		if char in word_set:
# 			return False
# 		word_set.add(char)
# 	return True
#
# #reverse c style string
# def one_two(string):
# 	arr = []
# 	for i in range(len(string)-2,-1,-1):
# 		arr.append(string[i])
# 	return ''.join(arr)
#
# #remove duplicates no extra buffer
# #const memory = ascii
# def one_three(stringArr):
# 	charArr = [0]*26
# 	for char in stringArr:
# 		charArr[ord(charArr)-97] += 1
# 	for i in range(len(stringArr)):
# 		if not charArr[ord(stringArr[i])] > 1:
# 			stringArr.pop(i)
# 	return stringArr
#
# #check if two strings are anagrams
# def one_four(string1, string2):
# 	string_one_dic = {}
# 	string_two_dic = {}
# 	for char in string1:
# 		string_one_dic.setdefault(char,0)
# 		string_one_dic[char] += 1
# 	for char in string2:
# 		string_two_dic.setdefault(char,0)
# 		string_two_dic[char] += 1
# 	if len(string_one_dic) != len(string_two_dic):
# 		return False
# 	for key in string_one_dic:
# 		if string_one_dic[key] != string_two_dic[key]:
# 			return False
# 	return True
#
# def one_six(matrix):
# 	n = len(matrix[0])
# 	'''
# 		[1][1][1][1][1]
# 		[2][2][2][2][2]
# 		[3][3][3][3][3]
# 		[4][4][4][4][4]
# 		[5][5][5][5][5]
#
# 		k1,...,kn
# 		top row is ks
# 		swap ks with right row
# 		swap ks with bottom row
# 		swap ks with left row
#
# 		[5][4][3][2][1]
# 		[5][4][3][2][1]
# 		[5][4][3][2][1]
# 		[5][4][3][2][1]
# 		[5][4][3][2][1]
# 		ks = 1,1,1,1,1
# 		ks = 1,2,3,4,5
# 		ks = 5,5,5,5,5
# 		ks = 5,4,3,2,1
#
# 		shrink left and right side of k
# 		ks = 2,2,2,
# 		k 2,3,4
# 		ks = 2,4,4,
# 		ks = 4,3,2
#
# 	'''
# 	col, row = n,n
# 	s,t = 0,0
#
# 	while (col-s)>2 and (row-t)>2:
# 		ks = []*(col-s)
# 		for i in range(s,col):
# 			ks.append(matrix[s][i])
#
#
# 		for i in range(j,row):
# 			ks[i], matrix[i][j] = matrix[i][j], ks[i]
# 		for i in range(col-1,s-1,-1):
# 			ks[i], matrix[col-1][i] = matrix[col-1][i], ks[i]
# 		for i in range(row-1,j-1,-1):
# 			ks[i], matrix[i][row-1] = matrix[i][row-1], ks[i]
# 		for i in range(s,col):
# 			ks[i], matrix[s][i] = matrix[s][i], ks[i]
#
# 	return matrix
#
# #	WRONG, ONLY NEED FLAGS FOR ROWS AND COLS, NOT ALL 2D SPACE, REDO
# def one_seven(matrix):
# 	m = len(matrix[0])
# 	n = len(matrix)
# 	zeroes = set()
# 	for i in range(m):
# 		for j in range(n):
# 			if matrix[i][j] == 0:
# 				zeroes.add((i,j))
# 				for k in range(m):
# 					zeroes.add((k,j))
# 				for l in range(n):
# 					zeroes.add((i,l))
#
# 	for i in range(m):
# 		for j in range(n):
# 			if (i,j) in zeroes:
# 				matrix[i][j] = 0
# 	return matrix

#rotate is substring
def one_eight(stringOne, stringTwo):
	#find start, dont find then ignore. then check is behind is substring?
	if len(stringOne) != len(stringTwo):
		return False
	start = False
	j = 0
	k = 0
	for i in range(len(stringOne)):
		if stringTwo[i] == stringOne[j]:
			if j == 0:
				k = i
			start = True
			j += 1
		else:
			start = False
			j = 0
	if start == False:
		return False
	if stringOne[k:] in stringTwo:
		return True
	if stringOne[:k] in stringTwo:
		return True
	return False

#LESSON, think of concatenating strings for similar strings
def one_eight_revised(stringOne, stringTwo):
	if len(stringOne) != len(stringTwo):
		return False
	return stringTwo in stringOne+stringOne

print(one_eight_revised('erbottlewat','waterbottle'))
