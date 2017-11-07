'''
selection review
very similar to quicksort in terms of set up.
no need to sort to find kth element in a list
but instead can be done in o(n)
quick sort can be o(nlogn) if we choose median
instead of pivot

tips:
raise value error for bad index not in between 0 <= k < n
basecase of n <=1 --> return arr[0]
use L, E, G
if k < len(L):
	select(L, k)
if k < select(len(L) + len(E)):
	return pivot
else:
	select(G, k - len(l)-len(E))

O(n) runtime

n + n / 2 + n / 4 + n / 8 + n / 16 + ... = n (1 + 1/2 + 1/4 + 1/8 + ...)
= 2n on average
worst case is 0(n^2) like quick sort if you pick the worst each
time
'''
import random

def select(arr, k):
	n = len(arr)
	if not 0 <= k < n:
		raise ValueError('not valid index in array')
	if n <= 1:
		return arr[0]
	pivot = random.choice(arr)
	L, E, G = [],[],[]
	for data in arr:
		if data < pivot:
			L.append(data)
		elif data == pivot:
			E.append(pivot)
		else:
			G.append(data)
	if k < len(L):
		return select(L, k)
	elif k < (len(L) + len(E)):
		return pivot
	else:
		return select(G, k - (len(L) + len(E)))

x = [1,2,3,4,5,6,7,8,9,10]
print(select(x,3))
