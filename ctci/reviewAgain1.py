from pprint import pprint
import random
'''
1.1: unique words in array
    -> dictionary, o(n) space, o(n) algo,
    -> can manipulate into set, do that
    -> sort the arr if need to still be arr,

1.2: reverse c string
create new arr for string, go up to the end, then go backwards when putting it in

1.3: remove duplicate without buffer
    -> const mem with set of ascii chars

1.4: if 2 strings are anagrams
    -> sort words then compare O(nlogn)
    -> const arr of ascii char, use ord to get right placement, then increment by 1
    start at 0, (or use dic), then compare and -= 1, if -1, return False, check at the end
    if all are 0
    -> or do for both, then values and compare and if key not in string2, then wrong

1.5: replace all %20,
    -> arr for string, then convert at the end
    -> loop string, when string meets at %20, seperate while loop, use it for skipping,
    keep going through loop

1.7: make row and col all 0s when find 0 in mxn matrix
    -> keep set of rows that have 0
    -> keep set of cols that have 0
    then loop and add to rows/cols
    then go through values in rows and cols and set 0

'''
def replaceCharSeq(word):
    n = len(word)
    i = 0
    output = []
    while i < n:
        if word[i] == '%' and i+2 < n:
            if word[i] == '%' and word[i+1] == '2' and word[i+2] == '0':
                i += 3
            else:
                output.append(word[i])
                i += 1
        else:
            output.append(word[i])
            i += 1
    return ''.join(output)

#Write an algorithm such that if an element in an MxN matrix is 0, its entire row and
#column is set to 0.

def one_seven(matrix):
    row = len(matrix)
    col = len(matrix[0])
    row_zeros = set()
    col_zeros = set()
    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 0:
                row_zeros.add(i)
                col_zeros.add(j)
    for i in range(row):
        for j in range(col):
            if i in row_zeros or j in col_zeros:
                matrix[i][j] = 0
    # for index in row_zeros:
    #     for j in range(col):
    #         matrix[index][j] = 0
    # for index in col_zeros:
    #     for i in range(row):
    #         matrix[i][index] = 0
    return matrix

matrix = [ [random.randrange(10) for i in range(10)] for i in range(10)]
# pprint(matrix)
#
#
# pprint(one_seven(matrix))

'''
2.1: remove duplicates in linkedlist
    -> loops? good question ask, perhaps put in node value in set?
    -> use buffer then keep data in set, if it is in there, then get out

    no buffer: two pointers, one is current, the other is runner which looks for duplicates
    prior and removes them, only has to remove one dup then break loop

2.2: go to kth from last
    go up point k forward with pointer 1,
    then while pointer1.next not null, have pointer2 go up and pointer1 go up
    return pointer2

2.3: remove middle node
    a->b->c->d->e
    temp = c.next = d
    if not temp:
        c = null
    temp_next = temp.next = e
    c.data = temp.data
    c.next = temp_next
    garbage collector

2.4:


'''

'''
3.1: arr of stacks, split arr into 3 sizes, increase into new arr of sizes

3.2: stack with min, give min attribute, when size 1 give min to first,
    compare with min each time you add.
    when removing element, ...
    how about have arr of mins, each time when inserting, append min to end of arr by comparing
    last min to new data
    then pop when. basically stack of mins

3.3: set of stacks
    keep size variable at 0, when adding size += 1, capacity is defined at 10,
    when size reaches 10, size = 0
    add new stack to set of stacks, repeat?

3.4: persistent stack
    prev on each node

3.5: queue with 2 stacks
    enqueue put in stack 1
    dequeue if stack 2 is empty, empty stack 1 into stack 2
    then pop stack
    if not empty, just pop stack

3.6: sort stack
    create stack2, this is what we return,
    while stack1 not empty
    ele = stack1.pop
    while stack2 not empty and stack2.peek > ele:
        stack1.push(stack2.pop)
    stack2.push(ele)
    return stack2

'''



'''

4.1: check if binary:
    check min and max depth,


4.2: route between two nodes:
    dfs,
    store queue in root: while root not empty, check children, if end break
    else add to queue

4.3: sorted arr to minimal height binary tree? maybe binary search tree?
    insert middle ele,
    then call recursively the left side for left child, right side for right children
    if low > high:
        return None

4.4: says binary tree, d linkedlist for each depth?
    i dont think binary matters,
    instead, need to keep track
    output = [new linkedlist]
    current_queue = queue
    next_queue = queue
    current_queue.add(root)
    while current_queue not empty:
        node = queue.dequeue
        output[-1].add(queue)
        next_queue.add(left, right children)
        if current_queue is empty:
            if next_queue not empty:
                current_queue = next_queue
                output.append(new linkedlist)

4.5: in order successort
    scenarios:
    have two children
        then it is right, then most left
    have only left
        check if it has parent, then it is parent, if no parent then no in order
    have only right
        then it is right, then most left
    have no parent
        need to check children
        only left then issue
        only right then ok

4.6: only works with reference to parent
    then go up and check?

    no reference to parent,
        search for val in left, for other search for val in right
        if both true, then current node is ancestor
        go all left or all right recursive,

4.7: big tree and small tree is small tree in big tree
    put in order and then check if have space
    else:
    have to check at each child can return False at early stages, if don't have memory

4.8: memory space of n**2
    but what you do instead is you keep track of what is behind you.
    making copys, and then going backwards all the way and if make sum, print path

'''


'''
8.1: fib smart look below

8.2:

'''

def fib_smart(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    x,y = 0,1
    for i in range(1,n):
        x,y = y,y+x
    return y

'''

9.1:



'''

def merge(a,b):
    a_length = len(a)
    b_length = len(b)
    a_filled_length = len(a)-len(b)
    for i in range(a_length-1,-1,-1):
        if a_filled_length > 0 and b_length > 0:
            if a[a_filled_length-1] <= b[b_length-1]:
                a[i] = b[b_length-1]
                b_length -= 1
            else:
                a[i] = a[a_filled_length-1]
                a_filled_length -= 1
    for i in range(b_length-1,-1,-1):
        a[i] = b[i]
    return a

# print(merge([1,3,6,7,9,None,None,None,None,None],[0,2,3,6,10]))

def select(arr,k):
    if not 0 <= k < len(arr):
        raise ValueError('not in bounds of arr')
    pivot = random.choice(arr)
    L,E,G = [],[],[]
    for x in arr:
        if x < pivot:
            L.append(x)
        elif x == pivot:
            E.append(pivot)
        else:
            G.append(x)
    if k < len(L):
        return select(L,k)
    elif k < len(L) + len(E):
        return pivot
    else:
        return select(G,k-len(L)-len(E))

# print(select([15,16,19,20,25,1,3,4,5,7,10,14],9))
bin_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
def binarySearch(arr,data):
    low = 0
    high = len(arr)-1
    while low <= high:
        mid = (low+high)//2
        if data == arr[mid]:
            return data
        elif arr[mid] > data:
            high = mid-1
        else:
            low = mid+1
    return False



def reverseString(string, first, last):
    while first < last:
        string[first], string[last] = string[last], string[first]
        first += 1
        last -= 1
    return string

def reverseWords(sentence):
    characters = list(sentence)
    reverseString(characters,0,len(characters)-1)

    first = last = 0
    while first < len(characters) and last < len(characters):
        if characters[last] == ' ':
            reverseString(characters,first,last-1)
            first = last+1
        last += 1
    if first < last:
        reverseString(characters,first,len(characters)-1)
    return ''.join(characters)

print(reverseWords('There are k lists of sorted integers'))









































# print(binarySearch(bin_arr,17))
