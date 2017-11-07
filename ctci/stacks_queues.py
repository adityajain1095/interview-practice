#3.2
#implement stack that also has min func
#keep track of min on each node for all beneath them
#compare min prev, and min you are now
#or keep mins in a seperate stack

# class Node(object):

# 	def __init__(self, data):
# 		self.data = data
# 		self.next = None
# 		self.prev = None

# class Stack(object):

# 	def __init__(self):
# 		self.top = None
# 		self.min = None
# 		self.size = 0

# 	def push(self, data):
# 		new_top = Node(data)
# 		new_top.next = self.top
# 		self.top = new_top
# 		self.size += 1
# 		if not self.min:
# 			self.min = self.top
# 		else:
# 			if self.top.data <= self.min:
# 				self.min = self.top

# 	def pop(self):
# 		if self.size:
# 			temp = self.top
# 			self.top = self.top.next
# 			self.size -= 1
# 			if self.size == 0:
# 				self.min = None
# 			elif self.min == temp:
# 				temp2 = temp.next
# 				_min = float('inf')
# 				while not temp2:
# 					if _min <= temp2.data:
# 						self.min = temp2
# 						_min = temp2.data
# 					temp2 = temp2.next
# 			return temp.data
# 		return None

# 	def min(self):
# 		if self.size:
# 			return self.min.data
# 		return None

#persistent stack 
'''
		 6-> 7 -> 8
		/
1->2->3->4->5->.....

s3 = s2.push(3)
s4 = s3.push(4)
.
.
.
s6 = s3.push(6)

make Node class have prev
return the top like stack implementation of stack
on the node, give it top attribute. can't have stack attribute

push(data):
	new_node = Node(data)
	new_node.prev = self
return new_node

pop():
top = .prev
return top

run example:

1.) 1, return at 1 
2.) 1->2 return at 2->1
3.) 1->2->3 return at 3->2->1
4.) 1->2->3 do this from step 2, can't have top attribute
	    \>4 return at 4->2->1
'''

#3.6
# sort stack in ascending order 1,2,3,4,5,6

'''



2
        5
7       4
1       4
4       3 
'''

def sort_stack(stack):

	output = []

	while stack:
		if output:
			if output[-1] <= stack[-1]:
				output.append(stack.pop())
			else:
				temp = stack[-1]
				stack.pop()
				while output[-1] >= temp:
					stack.append(output.pop())
					if not output:
						break
				output.append(temp)
		else:
			output.append(stack.pop())

	return output

'''
3

2        9
4        7        
1        5

		9
2		7
4		5
1		3
'''

print(sort_stack([1,4,2,3,9,7,5,20,15,12,0,4,50,30,20,20,33,43,3,17]))