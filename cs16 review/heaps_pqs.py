heaps_pqs.py

'''

priority queues and heaps
order is based on priority being on top, then next priority

upheap, put in last element, then go up and swap parent if less
remove min, swap with last element. then remove last element
then downheap from min and swap with lesser of two children
until can't

array based implementation: node at index i, left child index 2i
right child index 2i+1, ignore index 0

deque implementation of keeping track of binary left complete tree
when you add, you have basecase for root
then you can have deque, check if front has left
if it doesn't then add left to the end of deque
if it does, add right to the end of deque and remove 
front of deque

INSERT

		1

	2		3	

4	    5      6		7

[1]

[1,2,3] —> [2,3]

[2,3,4,5] —> [3,4,5] —> [3,4,5,6,7] —> [4,5,6,7]

DELETE

take end of deque, check if it is right, if it is add
parent to the front of deque, delete end of deque
else, delete end of dequeue

[4,5,6,7] --> [3,4,5,6,7] --> [3,4,5,6] -- > [3,4,5]
--> [2,3,4,5] --> [2,3,4] --> [2,3] --> etc...




'''