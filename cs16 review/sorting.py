'''

sorting algorithm review

selection sort--> usually inplace and iterative O(n^2), keep track of index or else it won't work
insertion sort --> o(n^2) but can be O(n) when relatively sorted

python sort --> it is timsort, mix of insertion and mergersort

mergesort and insertion can never truely be in place

id(object) --> returns unique id pointer address of object

'''
from math import log10
from random import randint

import random
import timeit
import numpy as np

#devide and conquer strategies:
'''
mergesort: split data into chunks of 1/2. then for each chunk,
merge then add them together with the other chunk you split from
then return final. attempt to do it inplace because copying makes it 0n^2
'''

def mergeSort(arr, low, high):
	if low >= high:
		return [arr[low]]
	mid = (low + high)//2
	left = mergeSort(arr, low, mid-1)
	right = mergeSort(arr, mid+1, high)
	return merge(left, right)

def merge(A, B):
	output = []
	lenA = len(A)
	lenB = len(B)
	a = 0
	b = 0
	while a < lenA and b < lenB:
		if A[a] <= B[b]:
			output.append(A[a])
			a+=1
		else:
			output.append(B[b])
			b+=1
	if a < lenA:
		for i in range(a,lenA):
			output.append(A[i])
	if b < lenB:
		for i in range(b,lenB):
			output.append(B[i])
	return output

arr = [1,10,2,9,3,8,4,7,5,6]
print(mergeSort(arr,0,len(arr)-1))

def mergeSort2(arr):
	n = len(arr)
	if n <= 1:
		return arr
	mid = n//2
	left = mergeSort2(arr[0:mid])
	right = mergeSort2(arr[mid:n])
	return merge2(left, right)

def merge2(A, B):
	output = []
	lenA = len(A)
	lenB = len(B)
	a = 0
	b = 0
	while a < lenA and b < lenB:
		if A[a] <= B[b]:
			output.append(A[a])
			a+=1
		else:
			output.append(B[b])
			b+=1
	if a < lenA:
		for i in range(a,lenA):
			output.append(A[i])
	if b < lenB:
		for i in range(b,lenB):
			output.append(B[i])
	return output

print(mergeSort2(arr))
'''
quicksort
'''

#not in place
def quicksort(arr):
	n = len(arr)
	if n <= 1:
		return arr
	pivot = random.choice(arr)
	L, E, G = [], [], []
	for i in range(n):
		if arr[i] < pivot:
			L.append(arr[i])
		elif arr[i] == pivot:
			E.append(arr[i])
		else:
			G.append(arr[i])
	return quicksort(L) + E + quicksort(G)

#inplace quicksort
'''
logic behind inplace:

base case, if end - start < 1:
	return
	if start < end:
		do all of it
#basically if end and start are the same

quicksort(takes in start and end)
get random pivot index
pass array, start, end and random pivot index in sub_partition
swap arr[pivot] with arr[start] so pivot at front of arr
from there have j and i be + 1 from start
[pivot, [j,i], the rest]
from here while j <= i:
if less than pivot, swap pivot[i] and pivot[j] and i+=1 always j+=1
switch pivot with arr[i-1] so now pivot in middle
and partitions value less than pivot and values greater than pivot

after array is partitioned. you want to call the same quicksort
on both sides of partition, but you can ignore the pivot point

so quicksort(array, start, pivot-1)
	quicksort(array, pivot+1, end)

basecase makes sense now with the above recursive statement
'''
def sub_partition(array, start, end, idx_pivot):

    'returns the position where the pivot winds up'

    if not (start <= idx_pivot <= end):
        raise ValueError('idx pivot must be between start and end')

    array[start], array[idx_pivot] = array[idx_pivot], array[start]
    pivot = array[start]
    i = start + 1
    j = start + 1
    while j <= end:
        if array[j] <= pivot:
            array[j], array[i] = array[i], array[j]
            i += 1
        j += 1

    array[start], array[i - 1] = array[i - 1], array[start]
    return i - 1

def quicksort2(array, start=0, end=None):

    if end is None:
        end = len(array) - 1

    if start < end:

	    idx_pivot = random.randint(start, end)
	    i = sub_partition(array, start, end, idx_pivot)
	    #print array, i, idx_pivot
	    quicksort2(array, start, i - 1)
	    quicksort2(array, i + 1, end)

def insertionSort(arr):
	n = len(arr)
	for i in range(1,n):
		if arr[i] < arr[i-1]:
			for j in range(i,0,-1):
				if arr[j] < arr[j-1]:
					arr[j], arr[j-1] = arr[j-1], arr[j]
				else:
					break
	return arr


def selectionSort(arr):
	n = len(arr)
	for i in range(n-2):
		_min = float('inf')
		_minIndex = n-1
		for j in range(i,n):
			if arr[j] <= _min:
				_minIndex = j
				_min = arr[j]
		arr[i], arr[_minIndex] = arr[_minIndex], arr[i]
	return arr

def selectionSort2(arr):
	n = len(arr)
	for i in range(n-2):
		_min = np.argmin(arr[i:n-1])
		arr[i], arr[_min] = arr[_min], arr[i]
	return arr

def radixSort(arr):
	pass


# a = 1
# b = a
# a = 2
# print(b)
#
# string = '''
# hello my name is
# petros dawit how are you
# '''
# print(string)
# string = (
# 	'hello my name is '
# 	'petros dawit how are you '
# )
# print(string)
#
# ss = '''\
# import random
# arr = [random.randrange(500) for i in range(500)]
# n = len(arr)
# for i in range(n-2):
# 	_min = float('inf')
# 	_minIndex = n-1
# 	for j in range(i,n):
# 		if arr[j] <= _min:
# 			_minIndex = j
# 			_min = arr[j]
# 	arr[i], arr[_minIndex] = arr[_minIndex], arr[i]
# '''
#
# ss2 = '''\
# import random
# import numpy as np
# arr = [random.randrange(500) for i in range(500)]
# n = len(arr)
# for i in range(n-2):
# 	_min = np.argmin(arr[i:n-1])
# 	arr[i], arr[_min] = arr[_min], arr[i]
# '''

# print(timeit.timeit(stmt=ss2, number=10000))
# arr = [random.randrange(100) for i in range(100)]
# print(arr)
# quicksort2(arr)
# print(arr)
# for i in range(10000):
# 	arr = [random.randrange(1000) for i in range(1000)]
# 	quicksort2(arr)

# Python program for implementation of Radix Sort

# A function to do counting sort of arr[] according to
# the digit represented by exp.
def countingSort(arr, exp1):

    n = len(arr)

    # The output array elements that will have sorted arr
    output = [0] * (n)

    # initialize count array as 0
    count = [0] * (10)

    # Store count of occurrences in count[]
    for i in range(0, n):
        index = (arr[i]/exp1)
        count[ (index)%10 ] += 1

    # Change count[i] so that count[i] now contains actual
    #  position of this digit in output array
    for i in range(1,10):
        count[i] += count[i-1]

    # Build the output array
    i = n-1
    while i>=0:
        index = (arr[i]/exp1)
        output[ count[ (index)%10 ] - 1] = arr[i]
        count[ (index)%10 ] -= 1
        i -= 1

    # Copying the output array to arr[],
    # so that arr now contains sorted numbers
    i = 0
    for i in range(0,len(arr)):
        arr[i] = output[i]

# Method to do Radix Sort
def radixsort(arr):

    # Find the maximum number to know number of digits
    max1 = max(arr)

    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1/exp > 0:
        countingSort(arr,exp)
        exp *= 10
    return arr

# print(radixsort([13, 8, 1992, 31, 3, 1993]))
# print(radixsort([10000, 1000, 100, 10, 0]))
# print(radixsort([0,001,0020,030,00005,006]))
# print(radixsort([-1,-2,-3,-4,-5]))
