from pprint import pprint

#4.1
#return true if balanced tree
'''

				 	A 4
			 B3              C1
			 check false      		 
		D 2
		check false      E 0     F0      G0      
	H 1    
J 0

SOLUTIONS: correct answer but extra memory with weights. instead can just
get mindepth and maxdepth and bool if > 1 for final answer.
'''

def four_one_main(root):
	weights = {}
	check = True
	four_one_helper(root, weights, check)
	return check 

def four_one_helper(node):
	if node.hasLeft():
		four_one_helper(node.left, weights, check)
	if node.hasRight():
		four_one_helper(node.right, weights, check)

	if node.hasRight and not node.hasLeft:
		weights[node] = node.right + 1
		if weights[node] >= 2:
			check = False

	if not node.hasRight and node.hasLeft:
		weights[node] = node.left + 1
		if weights[node] >= 2:
			check = False

	if node.hasRight and node.hasLeft:
		weights[node] = max(node.right+1,node.left+1)
		if (node.right - node.left) >= 2:
			check = False

	if not node.hasRight and not node.hasLeft:
		weights[node] = 0

#four_two
#directed graph, find if route for start and end

def four_two(graph, start, end):
	q = Queue()
	q.append(start)
	while q.size != 0:
		current = q.dequeue()
		nodes = current.neighbors()
		for node in nodes:
			if node == end:
				return True
			q.append(node)
	return False

#four_three
#Given a sorted (increasing order) array, write an algorithm to create a
#binary tree with minimal height.

'''

1,2,3,4,5,6,7,8,9,10

'''
class Node(object):

	def __init__(self,data):
		self.data = data
		self.right = None
		self.left = None

def four_three(arr):
	n = len(arr)
	root = four_three_helper(arr, 0, n-1)
	return root

def four_three_helper(arr, low, high):
	if low > high:
		return None
	mid = (high+low)/2
	#print(arr[mid],mid)
	node = Node(arr[mid])
	node.left = four_three_helper(arr,low,mid-1)
	node.right = four_three_helper(arr,mid+1,high)
	return node

#arr = [1,2,3,4]
#root = four_three(arr)
#print(root.data)
#print(root.left.data)
#print(root.right.data)
#print(root.right.right.data)


root = four_three([1,2,3,4,5,6,7,8,9,10])

'''
linkedlist for each node at each depth, use bfs, how to know which depth?
'''
def four_four(root):
	root_depth = 0
	lls = {}
	four_helper(root,root_depth,lls)
	return lls

def four_helper(node,depth,lls):
	if node_depth not in lls:
		lls[node_depth] = [node]
	else:
		lls[node_depth].append(node)
	if node.left:
		four_helper(node.left,node_depth+1,lls)
	if node.right:
		four_helper(node.right,node_depth+1,lls)

# lls = four_fou(root)
# pprint(lls)

	'''
	for bfs implementation:

		have results list, that takes in linkedlist of nodes
		start by adding root in first index of list
		then while true, (stop when you break)
		get size from last level in linkedlist in node. go through each child of node 
		and add to a new linkedlist you instantiated outside of loop. at the end, append
		to result list. if linkedlist.size == 0, break the loop
		return results
	'''

'''

in order successor, check for all cases. think of recursive definition
x.left, x, x.right, 
conditions:
node is none:
	return None
node.right is none and node.parent is none:
	return None
node.right is none but has parents:
	if parent.left = node:
		return None
	keep in loop for node = parent,
		if node.parent = None
		return None
		unless it has left
succesor keep most left when node.right exists

'''
def four_five(node):
	if not node:
		return None
	if not node.right and not node.parent:
		return None
	if not node.right:
		while node:
			parent = node.parent
			if parent.left == node:
				return node.parent
			else:
				node = node.parent
		return None
	succesor = node.right
	while succesor.left:
		succesor = succesor.left
	return succesor

'''
find common ancestor of two nodes from binary tree
'''
def four_six(node_one, node_two):
	# get depths of both,
	# if they are the same, keep checking,
	# if one is larger, go up large.size - small.size up.
	#then keep comparing and see if they are the same
	node_one_depth = four_six_helper(node_one)
	node_two_depth = four_six_helper(node_two)

	if node_one_depth == node_two_depth:
		while node_one_depth:
			if node_one == node_two:
				return node_one
			node_one = node_one.parent
			node_two = node_two.parent
			node_one_depth -= 1
		return None
	elif node_one_depth > node_two_depth:
		while node_one_depth != node_two_depth:
			node_one = node_one.parent
			node_one_depth -= 1

		while node_one_depth:
			if node_one == node_two:
				return node_one
			node_one = node_one.parent
			node_two = node_two.parent
			node_one_depth -= 1
		return None
	else:
		while node_one_depth != node_two_depth:
			node_two = node_two.parent
			node_two_depth -= 1

		while node_one_depth:
			if node_one == node_two:
				return node_one
			node_one = node_one.parent
			node_two = node_two.parent
			node_one_depth -= 1
		return None

def four_six_helper(node)
	c = 1
	while node.parent:
		c += 1
		node = node.parent
	return c

	'''
	test:

			1

		2		3

	5		4
6


	'''

'''
You have two very large binary trees: T1, with millions of nodes, and T2,
with hundreds of nodes. Create an algorithm to decide if T2 is a subtree of T1.
'''

def four_seven(t1, t2):
	#go recursively through each possible choice if memory is issue
	#string of preorder and inorder to check






