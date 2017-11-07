#chapter6 binary search

'''
dont do it in place, it is O(n)
instead use low and high
[0,1,2,3], 2.5, 0, 4
mid = 0 + 4 / 2 = 2
2.5 > arr[2] = 2
low = 3, high = 4
mid = 7/2 = 3

0.5 <
'''
#recursive inplace
def binarySearch(arr, k, low, high):
	mid = (low+high)//2
	n = len(arr)
	if n < 0:
		print('empty')
		return False
	if n == 1:
		if k == arr[0]:
			return True
		else:
			return False
	if low == high:
		if k == arr[low]:
			return True
		else:
			return False
	if k < arr[mid]:
		return binarySearch(arr, k, low, mid-1)
	elif k == arr[mid]:
		return True
	else:
		return binarySearch(arr, k, mid+1,high)
# '0.5'
# print(binarySearch([0.0,1.0,2.0,3.0], 0.5, 0, 4))
# '2.0'
# print(binarySearch([0.0,1.0,2.0,3.0], 2.0, 0, 4))
# '0.0'
# print(binarySearch([0.0,1.0,2.0,3.0], 0.0, 0, 4))
# '1.5'
# print(binarySearch([0.0,1.0,2.0,3.0], 1.5, 0, 4))
# '3.0'
# print(binarySearch([0.0,1.0,2.0,3.0], 3.0, 0, 4))

#iterative inplace

def binary(arr,k):
	low = 0
	high = len(arr)-1
	while low <= high:
		mid = (low + high)//2
		print(arr[mid])
		if k == arr[mid]:
			return True
		elif k < arr[mid]:
			high = mid-1
		else:
			low = mid+1
	return False

# print(binary([0,1,2,3,4,5,6,7,8,9,10],10))
import math
import string

def test(num):
	dic_letter = {}
	for i in range(1,27):
		dic_letter[i] = string.ascii_uppercase[i-1]
	output = []
	dic_letter[0] = 'Z'
	while num >= 0:
		num, index = divmod(num,26)
		print('num', num)
		print('index', index)
		print('dic_letter', dic_letter[index])
		num -=1
		output.append(dic_letter[index])
	print(output)
	l = ''.join(output[::-1])
	print(l)
	return l

test(52)
