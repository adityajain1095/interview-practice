'''

python linked list with node class

'''

class Node(object):

	def __init__(self, data):
		self.data = data
		self.next = None
		self.prev = None

class DLL(object):

	def __init__(self, node):
		self.head = node
		self.tail = node
		self.size = 1

	def remove(self, r):
		if self.size == 1:
			self.head = None
			self.tail = None
			self.size -= 1
			return r
		if self.head == r:
			self.size -= 1
			self.head = r.next
			return r
		if self.tail == r:
			self.size -= 1
			self.tail = r.prev
			self.tail.next = None
			return r
		node = self.head.next
		while node != None:
			if node == r:
				prev = node.prev
				next = r.next
				prev.next = next
				next.prev = prev
				break
			else:
				node = node.next

	def insert(self, node):
		self.tail.next = node
		temp = self.tail
		self.tail = node
		self.tail.prev = temp
		self.size += 1

	def printList(self):
		node = self.head
		string = ''
		while node != None:
			string += str(node.data) + ' '
			node = node.next
		print(string)

n1 = Node('A')
n2 = Node('B')
n3 = Node('C')
n4 = Node('D')
n5 = Node('C')
n6 = Node('D')
dll = DLL(n1)
dll.insert(n2)
dll.insert(n3)
dll.insert(n4)
dll.remove(n3)
dll.insert(n5)
dll.insert(n6)
dll.remove(n1)
dll.remove(n3)
dll.printList()