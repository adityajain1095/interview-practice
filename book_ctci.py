'''
strings
'''

from pprint import pprint

def check_permutation(string1=None,string2=None):
    if not string1 or not string2:
        return False

    if len(string1) != len(string2):
        return False

    dic_char = {}
    for char in string1:
        num = dic_char.get(char,0)
        dic_char[char] = num + 1

    for char in string2:
        num = dic_char.get(char,0)
        if num-1 < 0:
            return False
        dic_char[char] = num - 1

    return True

def urlify(string, n):
    string = list(string)
    full_n = 0
    for char in string:
        full_n += 1
    for i in range(n-1, -1, -1):
        if string[i] == ' ':
            string[full_n-1] = '0'
            string[full_n-2] = '2'
            string[full_n-3] = '%'
            full_n -= 3
        else:
            string[full_n-1] = string[i]
            full_n -= 1
    return ''.join(string)



x = 'Petros likes you    '
# print(list(x))
y = 16

# print(urlify(x,y))

def permutationPalindrome(string):
    dic_char = {}
    odd_count = 0
    delim = set([' ', '\n', '\t'])

    for char in string:
        if char not in delim:
            num = dic_char.get(char,0)
            dic_char[char] = num + 1

    pprint(dic_char)

    for char, value in dic_char.items():

        if value %2 == 1:
            odd_count += 1
    return odd_count < 2

# print(permutationPalindrome('tac c -- jj at'))

def one_edit_away(string1, string2):

    if abs(len(string1) - len(string2)) > 1:
        return False

    #3 cases O(n)


    return edits < 1

def palindrome_linkedlist(head):
    first_half, second_half = split(head)
    second_half = reverse(second_half)

    while first_half and second_half:
        print(first_half.data, second_half.data)
        if first_half.data != second_half.data:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True


def split(head):
    slow = head
    prev = None
    fast = head

    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next

    prev.next = None

    return head, slow

def reverse(head):
    prev = None
    while head:
        next = head.next
        head.next = prev
        prev = head
        head = next
    return prev


class ListNode(object):

    def __init__(self, data):
        self.data = data
        self.next = None

a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
d = ListNode(4)
e = ListNode(4)
f = ListNode(7)
g = ListNode(2)
h = ListNode(1)

a.next = b
b.next = c
c.next = d
d.next = e
e.next = f
f.next = g
g.next = h

def valid_bst(root):
    if not root:
        return True

    left = valid_bst(root.left)
    right = valid_bst(root.right)


    if root.left:
            if root.left.data >= root.data:
                left = False

    if root.right:
            if root.right.data <= root.data:
                right = False

    return left and right

class TreeNode(object):

    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def sortedArrayToBST(nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    return sortedArrayHelper(nums,0,len(nums)-1)

def sortedArrayHelper(nums,low,high):

    if low > high:
        return None

    if low == high:
        return TreeNode(nums[low])

    mid = (low+high)//2
    root = TreeNode(nums[mid])
    root.left = sortedArrayHelper(nums,low,mid-1)
    root.right = sortedArrayHelper(nums,mid+1,high)
    return root

nums = [1,2,3,4,5,6,7,8]
root = sortedArrayToBST(nums)
# print(valid_bst(root))

# print(palindrome_linkedlist(a))

def fib_fast(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    x, y = 0, 1
    for i in range(n):
        x, y = y, x+y
    return x

fib_arr = [0,1]

def fib_fast_mem(n):
    if n < len(fib_arr)-1:
        return fib_arr[n]
    for i in range(len(fib_arr)-1,n+1):
        fib_arr.append(fib_arr[-1]+fib_arr[-2])
    return fib_arr[n]

'''
0,1,1,2,3,5,8

'''
# print(fib_fast_mem(6))


def triple_step(n):
    if not n:
        return 0
    return triple_helper(0,n)

def triple_helper(jump,n):
    if jump > n:
        return 0
    if jump == n:
        return 1
    count = triple_helper(jump+1,n)
    count += triple_helper(jump+2,n)
    count += triple_helper(jump+3,n)
    return count

'''
1,1,1,1
1,2,1,1
1,1,2,1
2,2
2,1,1
1,3
3,1
'''

# print(triple_step(4))




'''

string explanation

'''

def one_one(string):
    dic_char = {}
    for char in string:
        if char in dic_char:
            return False
        dic_char[char] = True
    return True

def one_two(string1, string2):
    if len(string1) != len(string2):
        return False
    dic_char = {}
    for char in string1:
        dic_char[char] = dic_char.get(char,0) + 1
    for char in string2:
        dic_char[char] = dic_char.get(char,0) - 1
        if dic_char[char] < 0:
            return False
    return True

def one_three(string):
    new_string_length = 0
    for char in string:
        if char == ' ':
            new_string_length += 2
    new_string = list(string) + [' ']*new_string_length
    n = len(new_string)
    print(n)

    #actual code now starts here
    def urlify(new_string, n):
        k = 0
        for i in range(n-1,-1,-1):
            if new_string[i] != ' ':
                k = i
                break
        for i in range(k,-1,-1):
            if new_string[i] != ' ':
                new_string[n-1] = new_string[i]
                n -= 1
            else:
                new_string[n-1] = '0'
                new_string[n-2] = '2'
                new_string[n-3] = '%'
                n -= 3
        return new_string

    return urlify(new_string, n)

def one_four(string):
    dic_char = {}
    odd_count = 0
    delim = [' ','\t','\n']
    for char in string:
        if char not in delim:
            dic_char[char.lower()] = dic_char.get(char.lower(),0) + 1
            if dic_char[char.lower()] % 2 == 1:
                odd_count += 1
            else:
                odd_count -= 1
    return odd_count < 2

'hello'
'heallo'

def one_five(string1, string2):

    def insert_delete_check(string1, string2):
        check = 0
        j = 0
        for i in range(len(string2)):
            if string2[i] != string1[j]:
                check += 1
                if check > 1:
                    return False
            else:
                j += 1
        return True

    def replace_check(string1, string2):
        check = 0
        for i in range(len(string1)):
            if string1[i] != string2[i]:
                check += 1
                if check > 1:
                    return False
        return True

    if len(string1) == len(string2) + 1:
        return insert_delete_check(string2, string1)
    if len(string1) + 1 == len(string2):
        return insert_delete_check(string1, string2)
    if len(string1) == len(string2):
        return replace_check(string1, string2)
    return False



def one_six(string):
    if not string:
        return ''
    count = 1
    new_string = []
    for i in range(1,len(string)):
        if string[i] != string[i-1]:
            new_string.extend([string[i-1], str(count)])
            count = 1
        else:
            count += 1
    new_string.extend([string[-1], str(count)])
    return ''.join(new_string)

def one_seven(matrix):
    n = len(matrix)
    k = 0
    while k < n:
        for i in range(k,n-1-k):
            matrix[k][i], matrix[i][n-1-k], matrix[n-1-k][n-1-i], matrix[n-1-i][k] = matrix[n-1-i][k], matrix[k][i], matrix[i][n-1-k], matrix[n-1-k][n-1-i]
        k += 1
    return matrix

matrix0 = []
matrix1 = [1]
matrix2 = [[1,2],[3,4]]
matrix3 = [[1,2,3],[4,5,6],[7,8,9]]
matrix4 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

'''
1,2    3,1
3,4    4,2


1,2,3  7,2,1
4,5,6  4,5,6
7,8,9  9,8,3

1, 2, 3, 4       13,9,5,1
5, 6, 7, 8       14,10,6,2
9, 10,11,12      15,11,7,3
13,14,15,16      16,12,8,4
'''

def one_eight(matrix):
    row = len(matrix)
    col = len(matrix[0])
    col_0 = False
    row_0 = False
    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 0:
                if i == 0:
                    row_0 = True
                else:
                    matrix[i][0] = 0
                if j == 0:
                    col_0 = True
                else:
                    matrix[0][j] = 0

    for i in range(1, row):
        if matrix[i][0] == 0:
            for j in range(col):
                matrix[i][j] = 0

    for i in range(1,col):
        if matrix[0][i] == 0:
            for j in range(row):
                matrix[j][i] = 0

    if col_0:
        for j in range(row):
            matrix[j][0] = 0

    if row_0:
        for j in range(col):
            matrix[0][j] = 0
    return matrix

# matrix = [[1,1,1,1,0],[0,1,1,1,1],[1,1,0,1,1],[1,1,1,1,1]]
# pprint(matrix)
# print(one_eight(matrix))

'''

linkedlists

'''

class Node(object):

    def __init__(self, data):
        self.data = data
        self.next = None

def two_one(head):
    buff = set()
    node = head
    prev = None
    while node:
        if node.data not in buff:
            buff.add(node.data)
            prev = node
            node = node.next
        else:
            node = node.next
            prev.next = node
    return head

a = Node(3)
b = Node(1)
c = Node(6)
d = Node(3)
e = Node(7)
g = Node(5)
h = Node(4)
i = Node(5)
j = Node(3)
k = Node(10)

a.next = b
b.next = c
c.next = d
d.next = e
e.next = g
g.next = h
h.next = i
i.next = j
j.next = k

# head = two_one(a)
# print(head.data)

# print(head.next.data)
# print(head.next.next.data)
# print(head.next.next.next)

def two_two(head,k):
    fast_node = head
    while fast_node and k:
        fast_node = fast_node.next
        k -= 1
    if k:
        return None
    first_node = head
    while second_node:
        first_node = first_node.next
        second_node = second_node.next
    return first_node

def two_three(head):
    fast_node = head
    slow_node = head
    prev = None
    while fast_node and fast_node.next:
        fast_node = fast_node.next.next
        prev = slow_node
        slow_node = slow_node.next
    prev.next = slow_node.next
    return head

def two_four(head,partition):
    less_head, equal_head, greater_head = Node(0), Node(0), Node(0)
    less_head_start, equal_head_start, greater_head_start = less_head, equal_head, greater_head
    node = head
    while node:
        if node.data < partition:
            less_head.next = node
            less_head = less_head.next
        elif node.data == partition:
            equal_head.next = node
            equal_head = equal_head.next
        else:
            greater_head.next = node
            greater_head = greater_head.next
        node = node.next
    head = less_head_start.next
    less_head.next = equal_head_start.next
    equal_head.next = greater_head_start.next
    return head

def intersection(head1,head2):
    node1 , node2 = head1 , head2
    while node1 != node2:
        node1 = node1.next
        if not node1:
            node1 = head2
        node2 = node2.next
        if not node2:
            node2 = head1
    return node1 != None
# head = two_four(a,5)
# print(head.data)
# print(head.next.data)
# print(head.next.next.data)
# print(head.next.next.next.data)
# print(head.next.next.next.next.data)
# print(head.next.next.next.next.next.data)
# print(head.next.next.next.next.next.next.data)
# print(head.next.next.next.next.next.next.next.data)
# print(head.next.next.next.next.next.next.next.next.data)
# print(head.next.next.next.next.next.next.next.next.next.data)
# print(head.next.next.next.next.next.next.next.next.next.next)

def four_one(start, end):
    visited = set()
    visisted.add(start)
    q = [start]
    while len(q):
        node = q.pop()
        if node == end:
            return True
        for neighbor in get_neighbours(node):
            if neighbor not in visisted:
                visited.add(neighbor)
                q.append(neighbor)
    return False

class TreeNode(object):

    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None

def four_two(arr):
    if not arr:
        return None
    root = four_two_helper(arr,0,len(arr)-1)
    return root


def four_two_helper(arr, low, high):
    if low > high:
        return None
    mid = (low+high)//2
    node = TreeNode(arr[mid])
    left = four_two_helper(arr,low,mid-1)
    right = four_two_helper(arr,mid+1,high)
    return node

def four_three(root):
    if not root:
        return []
    list_of_depths = [[root]]
    while len(list_of_depths[-1]):
        depth_list = []
        for node in list_of_depths[-1]:
            if node.left:
                depth_list.append(node.left)
            if node.right:
                depth_list.append(node.right)
        list_of_depths.append(depth_list)
    list_of_depths.pop()
    return list_of_depths

def four_four(root):
    return four_four_max_helper(root) - four_four_min_helper(root) <= 1

def four_four_min_helper(root):
    if not root:
        return 0
    return min(four_four_min_helper(root.left), four_four_min_helper(root.right)) + 1

def four_four_max_helper(root):
    if not root:
        return 0
    return max(four_four_max_helper(root.left), four_four_max_helper(root.right)) + 1

def four_five(root):
    if not root:
        return True
    if root.left:
        if root.left > root:
            return False
    if root.right:
        if root.right.data < root:
            return False
    return four_five(root.left) and four_five(root.right)

def four_six(root):
    if not root:
        return None
    if root.right:
        node = root.right
        while node.left:
            node = node.left
        return node
    if root.parent:
        if root.parent.left == root:
            if root.parent.right:
                node = root.parent.right
                while node.left:
                    node = node.left
                return node
    return root.parent

def four_seven(root):
    pass

def four_eight(root, p, q):
    if not root:
        return None
    if root == p or root == q:
        return root
    left = four_eight(root.left,p,q)
    right = four_eight(root.right,p,q)
    if left and right:
        return root
    if left:
        return left
    if right:
        return right

def four_nine(root):
    pass

def eight_one(n):
    if n <= 0:
        raise ValueError('invalid input')
    if n == 1:
        return 0
    if n == 2:
        return 1
    if n == 3:
        return 2
    memo = [0]*n
    memo[1], memo[2], memo[3] = 1, 2, 4
    for i in range(4,n):
        memo[i] = memo[i-1] + memo[i-2] + memo[i-3]
    return memo[-1]

def eight_three(arr):
    low = 0
    high = len(arr)-1
    while low < high:
        mid = (low+high)//2
        if arr[mid] == mid:
            return mid
        if arr[mid] > mid:
            high = mid - 1
        else:
            low = mid + 1
    if arr[low] == low:
        return low
    return None

arr = [-5,-5,-5,-5,-5,3,6,10,12,14,16]
# print(eight_three(arr))

def eight_four(s):
    output = set()
    output.add('')
    s.sort()
    for x in s:
        temp = set()
        for y in output:
            temp.add(y+x)
        output.update(temp)
    return output

def eight_seven(string):
    output = []
    used = set()
    eight_seven_helper(string,output,used,[])
    return output

def eight_seven_helper(string,output,used,permutation):
    if len(permutation) == len(string):
        output.append(''.join(permutation))
    else:
        for i in range(len(string)):
            if i in used:
                continue
            if i > 0 and string[i] == string[i-1] and i-1 not in used:
                continue
            used.add(i)
            eight_seven_helper(string,output,used,permutation+[string[i]])
            used.remove(i)

def ten_three(arr,target):
    low, high = 0 , len(arr)-1
    while low < high:
        mid = (low+high)//2
        if arr[mid] == target:
            return True
        if target < arr[mid]:
            if arr[mid] > target and target >= arr[low]:
                high = mid-1
            else:
                low = mid+1
        else:
            if arr[high] >= target and target > arr[mid]:
                low = mid+1
            else:
                high = mid-1
    return arr[low] == target

# arr = [15,16,19,20,25,1,3,4,5,7,8,10,14]
# print(ten_three(arr,5))

def ten_ten(stream):
    pass
