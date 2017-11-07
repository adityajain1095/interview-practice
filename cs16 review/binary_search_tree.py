'''
binary search tree
definition: all left is lower, all right is higher

removing from a bst, 3 cases you have to deal with

remove leaf node: easy, just remove

remove internal node with one leaf, if this case, just switch
leaf with parent then remove parent

remove internal node with two leaves: in successor to take
can either be left or right. go down to one child, for our
case we will do in successor so go right. then take left
most child and replace data with what you want to remove
then remove the left most child

IMPORTANT NOTE FOR REMOVING IN BST, NEED REFERENCE TO
PARENT BECAUSE GARBAGE COLLECTOR WON'T GET RID OF VARIABLE
B/C YOU STILL HAVE REFERENCE TO NODE WITH OTHER NODES ABOVE
IT. NEED REFERENCE TO PARENT AND BREAK CONNECTIONS WITH CHILDREN
WHICH I DO IN REMOVE

'''

class Node(object):

	def __init__(self, data):
		self.data = data
		self.right = None
		self.left = None

	# def delete(self):
	# 	del self

class binarySearchTree(object):

	def __init__(self, root):
		self.root = Node(root)

	def search(self, value):
		if self.root:
			return self.searchHelper(self.root, value)
		else:
			print('tree is empty')
			return False

	def searchHelper(self, node, value):
		if node.data == value:
			return True
		if node.left:
			if value < node.data:
				return self.searchHelper(node.left, value)
		if node.right:
			if value > node.data:
				return self.searchHelper(node.right, value)
		return False

	def find(self, value):
		if self.root:
			return self.findHelper(None, self.root, value)
		else:
			print('tree is empty')
			return None, None

	def findHelper(self, parent, node, value):
		if node.data == value:
			return parent, node
		if node.left:
			if value < node.data:
				return self.findHelper(node, node.left, value)
		if node.right:
			if value > node.data:
				return self.findHelper(node, node.right, value)
		return None, None

	def insert(self, value):
		if self.root:
			node = Node(value)
			self.insertHelper(self.root, node)
		else:
			self.root = Node(value)

	def insertHelper(self, node, inserted):
		if node.data >= inserted.data:
			if not node.left:
				node.left = inserted
			else:
				self.insertHelper(node.left, inserted)
		else:
			if not node.right:
				node.right = inserted
			else:
				self.insertHelper(node.right, inserted)

	def remove(self, value):
		if self.root:
			parent, node = self.find(value)
			if node:
				if not node.left and not node.right:
					if parent.right:
						if parent.right == node:
							parent.right = None
							node = None
					if parent.left:
						if parent.left == node:
							parent.left = None
							node = None
				elif node.left and not node.right:
					node.data = node.left.data
					node.left = None
				elif node.right and not node.left:
					node.data = node.right.data
					node.right = None
				else:
					par, succ = self.inOrderSuccessor(node, node.right)
					succ.data, node.data = node.data, succ.data
					if not succ.left and not succ.right:
						if par.right:
							if par.right == succ:
								par.right = None
								succ = None
						if par.left:
							if par.left == succ:
								par.left = None
								succ = None
					elif succ.left and not succ.right:
						succ.data = succ.left.data
						succ.left = None
					elif succ.right and not succ.left:
						succ.data = succ.right.data
						succ.right = None
			else:
				print('value not in bt')
				return False
		else:
			print('tree is empty')
			return False

	def inOrderSuccessor(self, parent, node):
		if not node.left:
			return parent, node
		return self.inOrderSuccessor(node, node.left)

	def printTree(self, node):
		if node:
			self.printTree(node.left)
			print(str(node.data) + ' ')
			self.printTree(node.right)


print('binary search tree test')
bt = binarySearchTree(13)
bt.insert(8)
bt.insert(20)
# print('the root is ', bt.root.data)
# print('the root left is', bt.root.left.data)
# print('the root right is', bt.root.right.data)
# print('check if search works for root left', bt.search(8))
# print('check if search works for root right', bt.search(20))
# print('check if search works for root', bt.search(13))
# print('check if search works not in', bt.search(15))
bt.insert(5)
bt.insert(10)
bt.insert(9)
bt.insert(11)
# print('check if search works for 8 left', bt.search(5))
# print('check if search works for 8 right', bt.search(10))
# print('check if search works for 10 left', bt.search(9))
# print('check if search works for 10 right', bt.search(11))
# print('8 left', bt.root.left.left.data)
# print('8 right', bt.root.left.right.data)
# print('10 left', bt.root.left.right.left.data)
# print('10 right', bt.root.left.right.right.data)
bt.insert(17)
bt.insert(24)
# print('check if search works for 20 left', bt.search(17))
# print('check if search works for 20 right', bt.search(24))
# print('20 left', bt.root.right.left.data)
# print('20 right', bt.root.right.right.data)
# bt.printTree(bt.root)
# print('first case remove')
# bt.remove(8)
# print('check if search works for 8', bt.search(8))
# print(bt.root.left.right.left.data)
# bt.insert(10)
# print(bt.root.left.right.right.data)
# print(bt.root.left.right.right.right)
