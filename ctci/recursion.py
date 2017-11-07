from pprint import pprint
import random

#fib memoize
def eight_one(n):
	memo = [0]*(n+1)
	memo[0] = 0
	memo[1] = 1
	for i in range(2,n+1):
		memo[i] = memo[i-1] + memo[i-2]
	return memo[-1]
# print(eight_one(5))
# print(eight_one(24))

#how many paths for a robot in NxN grid
'''
you could also use stats and combinations to answer this

n choose k

n = options go right, down,
k = options go right

n!/(k!(n-k)!) for specific path, runtime is O(n)
for all the paths for each end, no end specified, this
way below is better
'''
def eight_two(n):
	if n < 1:
		return 'invalid input, n >= 1'
	matrix = [[0]*n for i in range(n)]
	pprint(matrix)
	#start is 0,0
	for k in range(1,n):
		matrix[0][k] = 1
		matrix[k][0] = 1
	pprint(matrix)
	for i in range(1,n):
		for j in range(1,n):
			matrix[i][j] = matrix[i-1][j]+matrix[i][j-1]

	pprint(matrix)
	paths = 0
	for i in range(n):
		for j in range(n):
			paths += matrix[i][j]

	return paths


#return all subsets of a set
def eight_three(my_set):
	result = [[]]
	for x in my_set:
		result.extend([y + [x] for y in result])
	return result



# print(len(eight_three((1,2,3,4,5,6))))


#print permutations of string
def eight_four(string):
	output = set()
	n = len(string)
	if n <= 1:
		return string
	for i in range(n):
		start = string[i]
		perm = eight_four(string[:i] + string[i+1:])
		for p in perm:
			output.add(start+p)
	return output


#all valid combinations of open and closed parentheses
#use list as string. because when switching from l to r,
# you need replace ( with ) at the index
# go with small experiment and figure out how to fix
def eight_five(n):
	output = set()
	string_arr = [''] * (n*2)
	eight_five_helper(n,n,output,string_arr, 0)
	return output

def eight_five_helper(l,r,output,string_arr, index):
	if l == 0 and r == 0:
		output.add(''.join(string_arr))
	if l > 0:
		string_arr[index] = '('
		eight_five_helper(l-1,r,output,string_arr, index + 1)
	if r > l:
		string_arr[index] = ')'
		eight_five_helper(l,r-1,output,string_arr, index + 1)




#paint fill function
#dont need visited, and also only look at previous color
def eight_six(color, k, l):

	colors = [1,2,3,4]
	row = 6
	col = 6
	matrix = [[0]*col for i in range(row)]
	for i in range(row):
		for j in range(col):
			matrix[i][j] = random.choice(colors)
	print(matrix)

	if k < row and 0 <= k and l < col and 0 <= l:
		prev = matrix[k][l]
		visited = set()
		eight_six_helper(color, k, l, row, col, matrix, prev, visited)
		return matrix
	else:
		return 'invalid inputs'

def eight_six_helper(color,k,l,row,col,matrix,prev, visited):
	if k < row and 0 <= k and l < col and 0 <= l:
		if matrix[k][l] == prev:
			matrix[k][l] = color
			if (k+1,l) not in visited:
				eight_six_helper(color, k+1, l, row, col, matrix, prev, visited)
			if (k-1,l) not in visited:
				eight_six_helper(color, k-1, l, row, col, matrix, prev, visited)
			if (k,l+1) not in visited:
				eight_six_helper(color, k, l+1, row, col, matrix, prev, visited)
			if (k,l-1) not in visited:
				eight_six_helper(color, k, l-1, row, col, matrix, prev, visited)

def eight_seven(n):
	memo = []
	if n < 25:
		memo = [0]*25
	else:
		memo = [0]*n
	memo[0] = 1
	memo[4] = 1
	memo[9] = 1
	memo[24] = 1
	for i in range(4):
		memo[i] = i + 1
	for i in range(5):
		memo[i+4] = 1 + i
	for i in range(5):
		memo[i+9] = 1 + i
	for i in range(5):
		memo[i+14] = 2 + i
	for i in range(5):
		memo[i+19] = 2 + i
	for i in range(25,n):
		memo[i] = min(memo[i-1]+1,memo[i-5]+1,memo[i-10]+1,memo[i-25]+1)
	return memo[n-1]

# print(eight_seven(100))
