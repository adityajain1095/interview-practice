'''
tree properites and traversals

nodes with parent-child relation

bfs --> start from the root, then visit children then grandchildren etc...
use array for parent/ left and right children

use i*2 and i*2+1 

'''

class Node(object):

	def __init__(self, data=None, parent=None):
		self.data = data
		self.parent = parent
		self.right = None
		self.left = None

	def hasParent(self):
		if self.parent:
			return True
		return False

	def getParent(self):
		return self.parent

	def getData(self):
		return self.data

	def changeData(self, data=None):
		self.data = data
		return self.data

	def hasRight(self):
		if self.right:
			return True
		return False

	def hasLeft(self):
		if self.left:
			return True
		return False

	def getRight(self):
		return self.right

	def getLeft(self):
		return self.left

	def makeRight(self, data=None):
		if data:
			self.right = Node(data, self)
			return self.right
		return None

	def makeLeft(self, data=None):
		if data:
			self.left = Node(data, self)
			return self.left
		return None

	def removeRight(self):
		self.right = None

	def removeLeft(self):
		self.left = None

class Tree(object):

	def __init__(self, rootData=None):
		self.size = 1
		self.root = Node(rootData)
		self.currentNode = None
		if self.root.getData():
			self.currentNode = self.root

	def getRoot(self):
		if self.root.data:
			return self.root
		return None

	def isEmpty(self):
		if self.size == 0:
			return True
		return False

	def addData(data=None):
		if self.root.getData():
			if data:
				if not self.currentNode.hasLeft():
					self.currentNode.makeLeft(data)
				if not self.currentNode.hasRight():
					self.currentNode.makeLeft(data)				

class binaryTree(object):

	def __init__(self, rootData, capacity=20):
		self.root = Node(rootData)
		if capacity < 20:
			capacity = 20
			print('by default we made your capacity 20')
		self.capacity = capacity
		self.array = [None]*capacity
		self.i = 0
		
	def addNode(self, data):
		if self.i == 0 and not self.root:
			self.root = Node(data)
		else:
			node = Node(data, self.array[int(self.i/2)])
			if self.i/2
			self.array[int(self.i/2)]
			self.i += 1
			if self.i >= self.capacity:
				capacity *= 2
				self.array += [None]*capacity

	def removeNode(self):
		if self.i != 0:
			self.array[self.i] = None
			self.array[int(self.i/2)].removeRight()
			self.i -= 1
			if self.i <= self.capacity/2:
				self.array = self.array[:capacity/2]
		else:
			self.root = None

# node1 = Node()
# print(node1.hasRight())
# print(node1.hasLeft())
# print(node1.changeData(5))
# print(node1.makeLeft(4).data)
# print(node1.makeRight(3).data)
# print(node1.right.parent.data)

tree = binaryTree('A', 10)
tree.addNode('B')
tree.addNode('C')
tree.addNode('D')
print(tree.root.data)
print(tree.capacity)
print(tree.array)
print(tree.root.left)
print(tree.root.right)
print(tree.root.left.left)
print(tree.root.left.right)
print(tree.root.right.left)
print(tree.root.right.left)















