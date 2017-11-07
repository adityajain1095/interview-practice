# class Node:
#     def __init__(self,data):
#         self.data = data
#         self.right = None
#         self.left = None
#
# class BinaryTree:
#     def __init__(self, root_node = None):
#         self.root = root_node
#
#     # Required collection modules have already been imported.
#     def number_of_full_nodes(self,root):
#         if not root:
#             return 0
#         return self.number_of_full_nodes_helper(root)
#
#     def number_of_full_nodes_helper(self,node):
#         fullNodes = 0
#         if node.left and node.right:
#             fullNodes += 1
#         if node.left:
#             fullNodes += self.number_of_full_nodes_helper(node.left)
#         if node.right:
#             fullNodes += self.number_of_full_nodes_helper(node.right)
#         return fullNodes
#
# root = Node(5)
# root.left = Node(6)
# root.right = Node(7)
# root.right.left = Node(3)
# root.right.right = Node(3)
# root.left.right = Node(4)
# root.left.right.left = Node(2)
# root.left.right.right = Node(3)
# bt = BinaryTree(Node(5))
# print(bt.number_of_full_nodes(root))

'''

[1,2,3]
[4,5,6]
[7,8,9]

[1,2,3]

[3,6,9]

[7,8,9]

[1,4,7]

[7,2,1]
[4,5,6]
[9,8,3]

top = 1

[7,4,1]
[8,5,2]
[9,6,3]

top = 2

STRINGS

unique chars
sort/dictionary

duplicate_char
dict counter, no buffer now:
    const dict of ascii counter

anagrams:
    sort them? --> no space if in place
    dic_counter --> no space if dict of ascii counter

replace_strings
count spaces
create arr same size + spaces
then when you see space, put it, counter

or list linkedlist, python list simple

rotate matrix
keep track of top (char) switch counter clockwise, fix top at the last one

set row and col

or set matrix[0][index] to 0 or matrix[index][0] to 0

for col,

for i in range(1,col):
    if matrix[i][0] == 0:
        set col to 0

for i in range(1,row):
    set row to 0

if [0][0] 0, set row to 0

if col_zero = 1, set col 0 to 0

LINKED LISTS

duplicates in list
runner/current and set

k from end
go up k with first pointer,
then second goes next and keep increment by 1 till reach end, second pointer is return

delete middle node,
swap data with next, if not next set none,
then set middle node next to its next next

keep adding simple

STACKS/QUEUES

min/ list of previous mins

sort stack in ascending order:
    two stacks
    return new stack
    while s not empty
        tmp = s.pop
        while r not empty and r.peek > temp
            s.push(r.pop)
        r.push(tmp)
    return r


TREE AND GRAPHS

depth check of left and right, ur depth is max+1, recursive

dfs, ids, bi directional search

min height binary tree, sorted list
    middle = root,
    left = (low, mid-1)
    right = (mid+1,high)
    if low > high:
        return None

linkedlist for different depths
bfs
list of levels and who is in it, iterate through previous level

in order successor
up once, then right once, if not return, if yes, return left loop
if not return, return right

all sums start anywhere

ids (if only goes down)


'''

def subsets(arr):
    output = ['']
    for x in arr:
        output.extend(x+y for y in output)
    return output

print(subsets(['1','2','3']))

def commonAncestor(root, node1, node2):
    if not root:
        return 0
    found = 0
    if root == node1 or root == node2:
        found += 1
    ret += commonAncestor(root.left,node1,node2)
    if ret == 2:
        return ret
    return found + commonAncestor(root,node1,node2)



class Node(object):

    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def printInorder(arr,root):
    if not root:
        return None
    printInorder(arr,root.left)
    arr.append(root.data)
    printInorder(arr,root.right)
    return arr

arr = [1,2,3,4,5,6,7,8,9,10]

def min_height_tree(arr):
    low, high = 0, len(arr)-1
    return min_helper(arr,low,high)

def min_helper(arr,low,high):
    if low > high:
        return None
    mid = (low+high)//2
    root = Node(arr[mid])
    root.left = min_helper(arr,low,mid-1)
    root.right = min_helper(arr,mid+1,high)
    return root

root = min_height_tree(arr)
print(root.data)
print(root.right.left.data)
print(root.right.right.data)

ar = []
print(printInorder(ar, root))
