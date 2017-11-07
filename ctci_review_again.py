'''
String:

all unique:
    set of ascii_lowercase, check if low in set, then remove, if not it, False, return True
reverse c style:
    last, first trick
remove duplicate char:
    dict of ascii_lowercase, all 0, see, make 1, if 1, pop character from lists
    return list join
anagrams check:
    dictionary of counts, first iteration
    second iteration, substract counts
replaces spaces %20:
    new_length_arr based off spaces
    go thru, when space, put in %20
rotate image 90 degrees (nxn matrix):

    [1 ,2 ,3, 4]
    [5 ,6 ,7 ,8]
    [9 ,10,11,12]
    [13,14,15,16]

set 0 matrix:
    column/row, keep track of rowlist and collist
    no memory, have row_o be [0,0], have col_0 be variable
    put first row has 0 when it is, put first col has 0 when it is
    go thru again, from 1 to n, ignore first col and first row, if 0, set
    col to 0/ row 0 when conditions, then do col setting to zero with row[0,0]
    check and col_0 check

isSubstring:
    concatenate 2 strings together, then check if s1 substring of s1+s2

'''
from pprint import pprint

matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

def rotate_90(matrix):
    n = len(matrix)
    k = 0

    while k < n:
        for i in range(k,n-1-k):
            top = matrix[k][i] # left
            right = matrix[i][n-1-k] #top
            bottom = matrix[n-1-k][n-1-i] # right
            left = matrix[n-1-i][k] # bottom
            matrix[k][i], matrix[i][n-1-k], matrix[n-1-k][n-1-i], matrix[n-1-i][k] = left, top, right, bottom
        k += 1
    return matrix

# print(rotate_90(matrix))

'''
linkedlist
remove dup from list:
    two pointers, for each next pointer, second pointer at head goes up and break
    and swap values to remove from linkedlist
nth from list,
    first goes up n, second = head, while first, first++ and second++, return second
delete middle
    did below
add_nums
    dummyNode, dummyReturn = dummy
    then return dummyReturn.next
    can do recursively
circular loop:
    when fast and slow meet again, if null, return None, then keep increment by 1, meet is loop

'''

def deleteMiddle(head):
    if not head or not head.next:
        return head
    slow = head
    fast = head
    prev = slow
    while fast and fast.next:
        prev = slow
        fast = fast.next.next
        slow = slow.next

    prev.next = slow.next
    return head

class ListNode(object):

    def __init__(self, data):
        self.data = data
        self.next = None

a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
d = ListNode(4)
e = ListNode(5)
f = ListNode(6)
# g = ListNode(7)
# h = ListNode(8)
a.next = b
b.next = c
c.next = d
d.next = e
e.next = f
# f.next = g
# g.next = h

x = deleteMiddle(a)
# print(x.data)
# print(x.next.data)
# print(x.next.next.data)
# print(x.next.next.next.data)
# print(x.next.next.next.next.data)
# print(x.next.next.next.next.next.data)
'''
1 2 3 4
5 6 7 8
9 101112
13141516

139 5 1
14106 2
15117 3
16128 4

'''

'''
stack and queues:

    sort stack in ascending order
    s1 is full, s2 is empty
    while s1 is not empty
        n = s1.pop,
        while !s2.empty s2.peek() > n:
            s1.push(s2.pop)
        s2.push(n)
    return s2

'''

'''

trees:
    balanced:
        if none:
            return 0
        left = recursively
        right = recursively
        if left == -1 right ==-1, abs(left-right) return -1
            return -1
        return 1+max(left,right)
    dag:
        use dfs,bfs, (mention ids, bi directional search)
    binary_tree_arr:
        dit it
    linkedlist of depths:
        output = [[root]]
        while output[-1]:
            x = []
            for i in range(len(output[-1])):
                add children to x in x
            output.append(x)
        output.pop()
        return output
    next_node:
        cases

    next commonAncestor:
        if not node:
            return False
        if node == p or node == q:
            return True
        left = recursively
        right = recursively
        return left or right
    all paths:
        start with root
        then children, minus sum
        until no longer possible? (nlogn) space and work


'''

def binary_tree_arr(arr,low,high):
    if low<high:
        mid = (low+high)//2
        node = ListNode(arr[mid])
        node.left = binary_tree_arr(arr,low,mid-1)
        node.right = binary_tree_arr(arr,mid+1,high)
        return node
    return ListNode(arr[low])

'''
sorting/searching
    merge 2, pretty simple
    sort anagrams:
        dict of anagrams,
    binary search on rotated sorted arr
        while low < high:
            mid =
            if a[mid] == target:
                return mid
            if a[mid] >= a[low]:
                if a[mid] > target and target >= a[low]:
                    high = mid-1
                else:
                    low = mid +1
            else:
                if target > a[mid] and a[high] >= target:
                    low= mid + 1
                else:
                    high = mid-1
        if a[low] == target:
            return low
        return None
    find_sorted_within_matrix:
        start right top, left is low, mid is high
        start



'''
from functools import cmp_to_key
from operator import itemgetter

def compare(x,y):
    ''.join(sorted(x))
    if ''.join(sorted(x)) > ''.join(sorted(y)):
        return 1
    if ''.join(sorted(x)) < ''.join(sorted(y)):
        return -1
    return 0

x = ['iceman', 'manice', 'abcded', 'maince']

x = [(65, 100), (70, 150), (56, 90), (75, 190), (60, 95), (68, 110)]
x.sort(key=itemgetter(0,1))

print(x)

# x.sort(key=cmp_to_key(compare))
# print(x)
