'''
strings
'''


from collections import *
from pprint import pprint

def unique_char(s):
    c = Counter(s)
    for char in s:
        if c[char] > 1:
            return False
    return True

def permutation_check(s1, s2):
    if len(s1) != len(s2):
        return False
    c = Counter(s1)
    for char in s2:
        if c[char] > 0:
            c[char] -= 1
        else:
            return False
    return True

def urlify(s, n):
    s = list(s)
    j = n-1
    i = len(s)-1
    while i >= 0:
        if s[j] == ' ':
            s[i] = '%'
            s[i-1] = '0'
            s[i-2] = '2'
            i -= 3
        else:
            s[i] = s[j]
            i -= 1
        j -= 1
    return ''.join(s)

def permutation(s):
    i, j = 0, len(s)-1
    while i < j:
        if s[i] == ' ':
            i += 1
        elif s[j] == ' ':
            j -= 1
        else:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
    return True

def string_compression(s):
    if not s:
        return 0
    output = []
    c = 1
    for i in range(1,len(s)):
        if s[i] == s[i-1]:
            c += 1
        else:
            output.extend([c,s[i-1]])
            c = 1
    output.extend([c,s[-1]])
    return ''.join(output)

def rotate_matrix(matrix):
    pprint(matrix)
    n = len(matrix)
    i = 0
    while i < n:
        for k in range(i,n-1):   
            matrix[i][k], matrix[k][n-1], matrix[n-1][n-1-k+i], matrix[n-1-k+i][i] = matrix[n-1-k+i][i], matrix[i][k], matrix[k][n-1], matrix[n-1][n-1-k+i]
        i += 1
        n -= 1        
    return matrix

def zero_matrix(matrix):
    row, col = len(matrix), len(matrix[0])
    col0 = False
    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 0:
                if j == 0:
                    col0 = True
                else:
                    matrix[0][j] = 0
                matrix[i][0] = 0

    for j in range(1,col):
        if not matrix[0][j]:
            for i in range(row):
                matrix[i][j] = 0
                
    for i in range(0,row):
        if not matrix[i][0]:
            for j in range(col):
                matrix[i][j] = 0
    
    if col0:
        for i in range(row):
            matrix[i][0] = 0
            
    return matrix

def spiral_matrix(matrix):
    row = len(matrix)
    col = len(matrix[0])
    
    row_i = 0
    col_i = 0
    output = []
    
    while row_i < row and col_i < col:
        # print(row_i,col_i)
        # print(row,col)
        for i in range(col_i,col):
            output.append(matrix[row_i][i])
        row_i += 1
        # print(output)
        for i in range(row_i,row):
            output.append(matrix[i][col-1])
        col -= 1
        # print(output)
        if col_i < col:
            for i in range(col-1,col_i-1,-1):
                output.append(matrix[row-1][i])
        row -= 1
        # print(output)
        if row_i < row:
            for i in range(row-1,row_i-1,-1):
                output.append(matrix[i][col_i])
        col_i += 1            
        # print(output)
    return output

matrix = [[1,2,3],[10,11,4],[9,12,5],[8,7,6]]
print(spiral_matrix(matrix))

# matrix = [[1,1,1,1],[0,1,1,1],[1,1,1,1]]
# print(zero_matrix(matrix))        


def sort_stacks(s):
    '''
    
    [1,2]
        ref = 4
    [7,5,4,2,1]
    
    [7,5,4,2,1]
    
    '''
    s2 = []
    while len(s):
        if not s2:
            s2.append(s.pop())
        else:
            node = s.pop()
            while s2 and node >= s2[-1]:
                s.append(s2.pop())
            s2.append(node)
    return s2

# print(sort_stacks([1,4,2,5,7]))
                

def paths_with_sum(root, val):
    paths_depth(root, 0, 0, val)
    
def paths_depth(root, count, current,val):
    if not root: 
        return 0, 0
    if not root.right or not root.left:
        if root.val == val:
            return 1, root.val
    leftCount, leftC   
                
def triple_step(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 4
    x,y,z = 4,2,1
    for i in range(4,n+1):
        temp_x = x
        x += y + z 
        y, z = temp_x, y       
    return x

def subsets(s):
    output = ['']
    for x in s:
        new_extend = []
        for i in range(len(output)):
            if i > 0 and output[i] == output[i-1]:
                continue
            new_extend.append(x+output[i])
        output.extend(new_extend)
    return output

# s = '112'
# print(subsets(s))

def permutations(s,output,path,used):
    # print(path)
    if len(path) == len(s):
        output.append(path)
    else:
        for i in range(len(s)):
            if i > 0 and s[i] == s[i-1] and i-1 not in used:
                break
            if i not in used:
                used.add(i)
                permutations(s,output,path+s[i],used)
                used.remove(i)
                

output = []
s = '112'
used = set()
l = list(s)
l.sort()
s = ''.join(l)
permutations(s,output,'',used)
# print(output)

class Node(object):
    
    def __init__(self, key):
        self.key = key
        self.next = None
        self.prev = None


class LRU(object):
    
    def __init__(self, capacity):
        self.head = None
        self.end = None

        self.capacity = capacity
        self.current_size = 0


    # PUBLIC


    def get(self, key):
        """
        :rtype: int
        """
        if key not in self.hash_map:
            return -1
        
        node = self.hash_map[key]

        # small optimization (1): just return the value if we are already looking at head
        if self.head == node:
            return node.value
        self.remove(node)
        self.set_head(node)
        return node.value
        

    def set(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: nothing
        """
        if key in self.hash_map:
            node = self.hash_map[key]
            node.value = value

            # small optimization (2): update pointers only if this is not head; otherwise return
            if self.head != node:
                self.remove(node)
                self.set_head(node)
        else:
            new_node = QNode(key, value)
            if self.current_size == self.capacity:
                del self.hash_map[self.end.key]
                self.remove(self.end)
            self.set_head(new_node)
            self.hash_map[key] = new_node


    # PRIVATE

    def set_head(self, node):
        if not self.head:
            self.head = node
            self.end = node
        else:
            node.prev = self.head
            self.head.next = node
            self.head = node
        self.current_size += 1


    def remove(self, node):
        if not self.head:
            return

        # removing the node from somewhere in the middle; update pointers
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

        # head = end = node
        if not node.next and not node.prev:
            self.head = None
            self.end = None

        # if the node we are removing is the one at the end, update the new end
        # also not completely necessary but set the new end's previous to be NULL
        if self.end == node:
            self.end = node.next
            self.end.prev = None
        self.current_size -= 1
        return node
        
    
def sum_lists(head1, head2):
    dummyNode = Node(0)
    head = dummyNode
    c = 0
    while head1.next and head2.next:
        n = Node((head1.val+head2.val+c)%10)
        c = (head1.val+head2.val)//10
        head1 = head1.next
        head2 = head2.next
        dummyNode.next = n
        dummyNode = dummyNode.next
    while head1.next:
        n = Node((head1.val+c)%10)
        c = (head1.val)//10
        head1 = head1.next
        dummyNode.next = n
        dummyNode = dummyNode.next    
    while head2.next:
        n = Node((head2.val+c)%10)
        c = (head2.val)//10
        head1 = head1.next
        dummyNode.next = n
        dummyNode = dummyNode.next         
    return head.next
    
def palindrome(head1):
    start, middle = split_in_half(head)
    middle = reverse_list(head)
    while start and middle:
        if start.val != middle.val:
            return False
        start, middle = start.next, middle.next
    return True

def split_in_half(head):
    slow = head
    fast = head
    while fast.next:
        slow = slow.next
        fast = fast.next.next
    return head, slow

def reverse_list(head):
    prev = None
    current = head
    while current:
        _next = current.next
        current.next = prev
        prev = current
        current = _next
    return prev

def partition(head,x):
    dummyBefore, dummyAfter = Node(0), Node(0)
    dummyBeforeHead, dummyAfterHead = dummyBefore, dummyAfter
    while head:
        if head.val <= x:
            dummyBefore.next = head
            dummyBefore = dummyBefore.next
        else:
            dummyAfter.next = head
            dummyAfter = dummyAfter.next
    dummyBefore.next = dummyAfterHead.next
    return dummyBeforeHead.next

def search(start,end):
    #dfs
    s = [start]
    visited = set()
    visited.add(start)
    while len(s):
        node = s.pop()
        if start == end:
            return True
        for neighbor in node.neighbors():
            if neighbor not in visited:
                s.append(neighbor)
                visited.add(neighbor)
    return False

def bidirectional_search(start, end):
    visited_end = set()
    visited_start = set()
    s1 = [start]
    s2 = [start]
    
    while len(s1) and len(s2):
        node1 = s1.pop()
        if node1 == end or node1 in visited_end:
            return True
        for neighbor in node1.neighbors():
            if neighbor not in visited_start:
                s1.append(neighbor)
                visited_start.add(neighbor)
        node2 = s2.pop()
        if node2 == start or node2 in visited_start:
            return True
        for neighbor in node2.neighbors():
            if neighbor not in visited_end:
                s2.append(neighbor)
                visited_end.add(neighbor)
    return False
                
def validate_bst(root):
    if not root:
        return True
    if root.left:
        if root.left.val > root.val:
            return False
    if root.right:
        if root.right.val <= root.val:
            return False
    return validate_bst(root.left, root.right)

def inorder_validate_bst(root):
    s = []
    current = root
    check = None
    while current or len(s):
        while current:
            s.append(current)
            current = current.left
        if check:
            if check.val > current.val:
                return False
        check = current        
        current = current.right
    return True

'''

3,5
5 full
3 full
5-> 2
empty 3
put 2 in 3
fill 5
put 1 until 3 full
4 in 5 now
done



'''


def triple_step(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 4
    x,y,z = 4,2,1
    for i in range(4,n+1):
        x, y, z = x+y+z,x,y
    return x
        
print(triple_step(5))

def subsets(s):
    s.sort()
    output = [[]]
    for x in s:
        print(x)
        print(output)
        for i in range(len(output)):
            if i > 0 and output[i] == output[i-1]:
                continue
            output.append([x]+output[i])
    return output

print(subsets([1,1,1,2]))
    