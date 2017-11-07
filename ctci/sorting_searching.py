from pprint import pprint
import random

'''

You are given two sorted arrays, A and B, and A has a large enough buffer at the end
to hold B. Write a method to merge B into A in sorted order.

merge algorithm but in place, make sure to keep track of indexes and placing,
printing is important

'''
def nine_one(A, B):
	n_a = len(A)
	n_b = len(B)
	k = n_a - 1
	a = 0
	b = 0
	i = k
	while i >= 0 and a < n_a-n_b and b < n_b:
		# print('i',i)
		# print('b value',b)
		# print('a value',a)
		# print('a index', k-n_b-a)
		# print('b index', n_b-1-b)
		if B[n_b-1-b] > A[k-n_b-a]:
			A[i] = B[n_b-b-1]
			b += 1
		else:
			A[i] = A[k-n_b-a]
			a += 1
		i -= 1
		print('A', A)
	while i >= 0 and b < n_b:
		# print('b open', b)
		# print('B prev value', B[n_b-b-1])
		# print('i', i)
		# print('index', n_b-b-1)
		# print('b close')
		A[i] = B[n_b-b-1]
		b += 1
		i -= 1
	return A

A = [-2,3,4,20,'','','','','']
B = [1,3,5,7,9]
# print(nine_one(A, B))


'''

Given a sorted array of n integers that has been rotated an unknown number of
times, give an O(log n) algorithm that finds an element in the array. You may assume
that the array was originally sorted in increasing order.
EXAMPLE:
Input: find 5 in array (15 16 19 20 25 1 3 4 5 7 10 14)
Output: 8 (the index of 5 in the array)

1 public static int search(int a[], int l, int u, int x) {
2 while (l <= u) {
3 int m = (l + u) / 2;
4 if (x == a[m]) {
5 return m;
6 } else if (a[l] <= a[m]) {
7 if (x > a[m]) {
8 l = m+1;
9 } else if (x >=a [l]) {
10 u = m-1;
11 } else {
12 l = m+1;
13 }
14 }
15 else if (x < a[m]) u = m-1;
16 else if (x <= a[u]) l = m+1;
17 else u = m - 1;
18 }
19 return -1;
20 }
21
22 public static int search(int a[], int x) {
23 return search(a, 0, a.length - 1, x);
24 }
'''

#doesn't work with duplicates

'''

big files of memory, break into k chunks then O(nlogn) sort, do it one by one so not
to be too intensive with memory

afterwards, merge one by one
WORK ON THIS AND REVIEW AGAIN

'''

def nine_five(arr,word):

	n = len(arr)
	low = 0
	high = n-1
	return nine_five_helper(arr,word,low,high)

def nine_five_helper(arr, word, low, high):
	mid = (low + high)/2
	i = 1
	j = 1
	while mid < high:
		if not arr[mid]:
			mid += 1
	if not arr[mid]:
		mid = (low + high)/2
		while mid < low:
			if not arr[mid]:
				mid -= 1
	if not arr[mid]:
		return

arr = ['at', '', '', '', 'ball', '', '', 'car', '', '', 'dad', '', '']
word = 'ball'

'''
BOOK ANSWER
public int search(String[] strings, String str, int first, int last) {
2 while (first <= last) {
3 // Ensure there is something at the end
4 while (first <= last && strings[last] == “”) {
5 --last;
6 }
7 if (last < first) {
8 return -1; // this block was empty, so fail
9 }
10 int mid = (last + first) >> 1;
11 while (strings[mid] == “”) {
12 ++mid; // will always find one
13 }
14 int r = strings[mid].compareTo(str);
15 if (r == 0) return mid;
16 if (r < 0) {
17 first = mid + 1;
18 } else {
19 last = mid - 1;
20 }
21 }
22 return -1;
23 }
'''

'''
Given a matrix in which each row and each column is sorted, write a method to find
an element in it.
'''
#check all scenarios
def nine_six(matrix, element):
	row = len(matrix)
	col = len(matrix[0])
	i = 0
	j = col - 1
	while i < row and j > -1:
		if element == matrix[i][j]:
			return matrix[i][j]
		if element >= matrix[i][j]:
			i += 1
		else:
			j -= 1
	return None

#double sort
arr = [(65, 100), (70, 150), (56, 90), (75, 190), (60, 95), (68, 110)]
def nine_seven(arr):
	if len(arr) <= 1:
		return arr
	pivot = random.choice(arr)
	L, E, G = [],[],[]
	for x in arr:
		if x[0] == pivot[0]:
			if x[1] == pivot[1]:
				E.append(x)
			elif x[1] < pivot[1]:
				L.append(x)
			else:
				G.append(x)
		elif x[0] < pivot[0]:
			L.append(x)
		else:
			G.append(x)
	#only do E[0] if you want unique
	return nine_seven(L) + [E[0]] + nine_seven(G)

pprint(nine_seven(arr))
