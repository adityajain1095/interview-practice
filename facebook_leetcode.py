
from collections import *
from pprint import pprint
from heapq import *


#two_sum
def two_sum(nums, target):
    num_dic = set()
    if target-x in num_dic and target-x != x:
        return [min(x,target-x),max(x,target-x)]
    num_dic.add(x)
    return [-1,-1]
    
#Regular Expression Matching 22.0% Hard
def isMatch(text, pattern):
    text = ' ' + text
    pattern = ' ' + pattern
    memo = [[False] * (len(pattern)) for i in range(len(text))]
    memo[0][0] = True
    
    for i in range(1,len(memo[0])):
        if pattern[i] == '*':
            memo[0][i] = memo[0][i-2]
            
    for i in range(1,len(memo)):
        for j in range(1,len(memo[0])):
            if text[i] == pattern[j] or pattern[j] == '.':
                memo[i][j] = memo[i-1][j-1]
            elif pattern[j] == '*':
                memo[i][j] = memo[i][j-2]
                if pattern[j-1] == text[i] or pattern[j-1] == '.':
                    memo[i][j] = max(memo[i-1][j], memo[i][j])
            else:
                memo[i][j] = False
    # pprint(memo)
    return memo[-1][-1]

lists = []
lists.append([4,10,15,24])
lists.append([0,9,12,20])
lists.append([5,18,22,30])


def shortest_range_k_lists(k, lists):
    max_value = max([arr[0] for arr in lists])
    min_heap = [(arr[0],i) for i, arr in enumerate(lists,0)]    
    heapify(min_heap)
    min_range = float('inf')
    
    is_empty = {}
    for i, arr in enumerate(lists,0):
        is_empty[i] = len(arr)
    
    while True:
        node = heappop(min_heap)
        min_value, index = node[0], node[1]
        is_empty[index] = is_empty[index]-1                    
        min_range = min(max_value - min_value, min_range)
        
        if not is_empty[index]:
            break
                
        new_insert = lists[index][len(lists[index])-is_empty[index]]
        max_value = max(max_value, new_insert)
        heappush(min_heap, (new_insert, index))
        
    return min_range

# print(shortest_range_k_lists(len(lists), lists))
        
def longest_substring_without_repeat(string):
    current_set = set()
    ans = i = j = 0
    while j < len(string):
        if string[j] not in current_set:
            current_set.add(string[j])
            j += 1
            ans = max(ans, j-i)
        else:
            current_set.remove(string[i])
            i += 1
    return ans
# 
# print(longest_substring_without_repeat("abcabcbb"))

def romanToInt(string):
    dic_roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    #200 -> 
    integer = 0
    for i in range(len(s)-1):
        if dic_roman[s[i]] > dic_roman[s[i+1]]:
            integer -= dic_roman[s[i]]
        else:
            integer += dic_roman[s[i]]
    integer += dic_roman[s[i]]
    return integer

def threeSum(nums):
    result = []
    nums.sort()
    for i in range(len(nums)-2):
        j, k = i+1, len(nums)-1
        if i > 0 and nums[i] == nums[i-1]:
            continue #no repeats
        while j < k:
            s = nums[i] + nums[j] + nums[k]
            if s > 0:
                j -= 1
            elif s < 0:
                k += 1
            else:
                result.append((nums[i],nums[j],nums[k]))
                while j < k and nums[j] == nums[j+1]:
                    j += 1 # no repeats
                while j < k and nums[k] == nums[k+1]:
                    k -= 1 #no repeats
                j += 1
                k -= 1
    return result

def letterCombinations(digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    if not digits:
        return []
    n = len(digits)
    result = []
    helper(result, '', 0, 0, digits)
    return result

def helper(result, path, target, index, digits):
    phone = {'0': [' '], '1':['*'], '2':['a','b','c'], '3':['d','e','f'], '4':['g','h','i'], '5':['j','k','l'], '6':['m','n','o'], '7':['p','q','r', 's'], '8':['t','u','v'], '9':['w','x', 'y', 'z']}
    if len(path) == len(digits):
        result.append(path)
    else:
        for i in range(len(phone[digits[index]])):
            helper(result, path+phone[digits[index]][i], target+1, index+1, digits)
                
def valid_parentheses(string):
    s = []
    for char in string:
        if char == '{' or char == '[' or char == '(':
            s.append(char)
        else:
            if len(s) == 0:
                return False
            if char == '{':
                if s.pop() != '}':
                    return False
            elif char == '[':
                if s.pop() != ']':
                    return False
            else:
                if s.pop() != ')':
                    return False
    return len(s) == 0
     
def mergeKlists(lists):
    res = []
    k = len(lists)
    index = 0
    min_heap = []
    for arr in lists:
        if not arr:
            continue
        min_heap.append((arr.val, arr))
        index += 1
    heapify(min_heap)
    while len(min_heap) > 0:     
        min_value, node = heappop(min_heap)
        res.append(min_value)
        if node.next:
            heappush(min_heap, (node.next.val, node.next))
    return res

class ListNode(object):
    
    def __init__(self, data):
        self.data = data
        self.next = None
        
a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
d = ListNode(4)
e = ListNode(5)
a.next = b
b.next = c
c.next = d
d.next = e

def reverse_nodes_in_k_groups(head, k):
    node = head
    previous_prev = None
    
    while node:
        i = 0
        current = node
        end = node
        while i<k and node:
            node = node.next
            i += 1
            
        if i == k and not node:
            prev = None
            while current:
                next = current.next
                current.next = prev
                prev = current
                current = next    
            return prev    
        
        if not node:
            break
        
        prev = node
        while current and i > 0:
            next = current.next
            current.next = prev
            prev = current
            current = next
            i -= 1
        if previous_prev:
            previous_prev.next = prev
            previous_prev = prev
        else:
            head = prev
            previous_prev = end

    return head    
    
# n1 = reverse_nodes_in_k_groups(a,5)
# print(n1.data)
# print(n1.next.data)
# print(n1.next.next.data)
# print(n1.next.next.next.data)
# print(n1.next.next.next.next.data)

def search_rotated(a, target):
    low, high = 0, len(a)-1
    while low < high:
        mid = (low+high)//2
        if a[mid] == target:
            return mid
        elif a[high] > a[mid]:
            if a[high] >= target and target > a[mid]:
                low = mid + 1
            else:
                high = mid - 1
        else:
            if a[mid] > target and target >= a[low]:
                high = mid - 1
            else:
                low = mid + 1
    if a[low] == target:
        return low
    return -1 

def search_rotated_dup(a, target):
    if not a:
        return False
    low, high = 0, len(a)-1
    while low <= high:
        mid = (low+high)//2
        if a[mid] == target:
            return True
        if a[low] == a[mid] and a[mid] == a[high]:
            low += 1
            high -= 1
        elif a[low] <= a[mid]:
            if a[mid] > target and target >= a[low]:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if a[high] >= target and target > a[mid]:
                low = mid + 1
            else:
                high = mid - 1
    return False

def count_and_say(n):
    if n == 1:
        return '1'
    integer = '1'
    for i in range(1,n):
        new_integer = ''
        c = 1
        for i in range(1,len(integer)):
            if integer[i] != integer[i-1]:
                new_integer += str(c) + integer[i-1]
                c = 1
            else:
                c += 1
        new_integer += str(c) + integer[-1]
        integer = new_integer
    return integer

# print(count_and_say(5))


def multiply_strings(num1, num2):
    res = [0]*(len(num1)+len(num2))
    for i in range(len(num1)-1,-1,-1):
        for j in range(len(num2)-1,-1,-1):
            res[i+j+1] += int(num1[i])*int(num2[j])
            res[i+j] += res[i+j+1]//10
            res[i+j+1] %= 10
    start = 0
    while start < len(res) and res[start] == 0:
        start += 1
    return ''.join(map(str,res[start:])) or '0'

# print(multiply_strings('0','0'))

def wildcard_matching(string, pattern):
    s = p = match = 0
    starIdx = -1
    while s < len(string):
        if p < len(pattern) and string[s] == pattern[p] or pattern[p] == '?':
            p, s = p+1, s+1
        elif p < len(pattern) and pattern[p] == '*':
            starIdx = p
            match = s
            p += 1
        elif starIdx != -1:
            p = starIdx+1
            match += 1
            s = match
        else:
            return False
    
    while p < len(pattern) and pattern[p] == '*':
        p += 1
    
    return p == len(pattern)

'''
concept:
if same += 1 for both
if * for p, keep track of current and one ahead
    then for next, if same then keep going += 1
    if not, set p back start + 1, s back to match 2
if new *, then update start idx,

match is for keeping track of what is next consecutive after last start id

aabcdabda
a?*d*ac

s = 2
p = 2
match = 0
star = -1

bcdabda
*d*ac

s = 5
p = 4
match = 4
star = 2

abac
*ac

s = 8
p = 7
match = 7
star = 4

'''
    
def groupAnagrams(arr):
    dic_arr = {}
    for word in arr:
        key = ''.join(sorted(word))
        if key not in dic_arr:
            dic_arr[key] = [word]
        else:
            dic_arr[key].append(word)
    res = []
    for key, value in dic_arr.items():
        res.append(value)
    return res
            
# print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))

def myPow(x,n):
    
    if not n:
        return 1
    if n < 0:
        return 1 / myPow(x, -n)
    if n%2:
        return x * myPow(x,n-1)
    return myPow(x*x, n/2)
                
# print(myPow(2,2))
    
# def insertRanges(intervals, newInterval):
#     left = []
#     right = []
#     for interval in intervals:
#         if interval.start < newInterval.start:
#             left.append(interval)
#         if interval.end > newInterval.end:
#             right.append(interval)    
#     leftUpdated = False
#     rightUpdated = False
#     if left:
#         if left[-1].end >= newInterval.start:
#             left[-1].end = max(left[-1].end, newInterval.end)
#             leftUpdated = True
#     if right:
#         if leftUpdated:
#             if right[0].start <= left[-1].end:
#                 last_left = left.pop()
#                 right[0].start = min(last_left.start,right[0].start)
#                 rightUpdated = True
#         else:
#             if right[0].start <= newInterval.end:
#                 right[0].start = min(newInterval.start,right[0].start)
#                 rightUpdated = True
# 
#     if rightUpdated or leftUpdated:
#         return left + right
#     return left + [newInterval] + right
#         
# intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]]
# newInterval = [4,9]
# print(insertRanges(intervals, newInterval))

def addBinary(a,b):
    carry = 0
    result = []
    i = len(a)-1
    j = len(b)-1
    while i >= 0 or j >= 0 or carry == 1:
        aDigit = bDigit = 0
        if i >= 0:
            aDigit = int(a[i])
        if j >= 0:
            bDigit = int(b[j])
        result.append(str(aDigit ^ bDigit ^ carry))
        if carry + aDigit + bDigit >= 2:
            carry = 1
        else:
            carry = 0
        i -= 1
        j -= 1
    return ''.join(result[::-1])




def sqrt(x):
    if not x:
        return 0
    left = 1 
    right = 1000000000
    while True:
        mid = (left+right)//2
        if mid > x//mid:
            right = mid - 1
        else:
            if mid + 1 > x//(mid + 1):
                return mid
            left = mid + 1 
    
def simplePath(path):
    path = [x for x in path.split('/') if x != '.' and x != '']
    s = []
    for x in path:
        if x != '..':
            s.append(x)
        else:
            if len(s) == 0 or s[-1] == '..':
                s.append(x)
            else:
                s.pop()
    return '/' + '/'.join(s)
                
# 
# path = "/.././../a/./b/../../c/"   
# print(simplePath(path))
    
def sortColors(nums):
    if not nums:
        return []
    j = 0
    k = len(nums)-1
    for i in range(len(nums)):
        if nums[i] == 0:
            nums[i], nums[j] = nums[j], nums[i]
            j += 1
    for i in range(len(nums)-1,-1,-1):
        if nums[i] == 2:
            nums[i], nums[k] = nums[k], nums[i]
            k -= 1
    return nums

def min_window(s, t):
    need, missing = Counter(t), len(t)
    i = I = J = 0
    for j, c in enumerate(s, 1):
        missing -= need[c] > 0
        need[c] -= 1
        if not missing:
            while i < j and need[s[i]] < 0:
                need[s[i]] += 1
                i += 1
            if not J or j - i <= J - I:
                I, J = i, j
    return s[I:J]

def subsets(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    nums.sort()
    result = []
    self.helper(nums, [], result, 0)
    return result
    
def helper(nums, path, result, index):
    result.append(path)        
    for i in range(index, len(nums)):
        if i != 0 and nums[i] == nums[i-1]:
            continue
        self.helper(nums,path+[nums[i]],result,i+1)
#         
def exist(board, word):
    """
    :type board: List[List[str]]
    :type word: str
    :rtype: bool
    """
    if not board:
        return False
    row = len(board)
    col = len(board[0])
    for i in range(row):
        for j in range(col):
            result = helper(i,j,row,col,board,0,word)
            if result:
                return True
    return False

def helper(i, j, row, col, board, index, word):
    if index == len(word):
        return True
    if i < row and j < col and i >= 0 and j >= 0:
        if board[i][j] == word[index]:
            temp = board[i][j]
            board[i][j] = '#'
            w = helper(i+1, j, row, col, board, index+1, word)
            x = helper(i-1, j, row, col, board, index+1, word)
            y = helper(i, j+1, row, col, board, index+1, word)
            z = helper(i, j-1, row, col, board, index+1, word)
            board[i][j] = temp
            return w or x or y or z
    return False
        
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCCED"

# print(exist(board, word))
def removeDuplicates(nums):
    i = 0
    for num in nums:
        if i < 2 or num != nums[i-2]:
            nums[i] = num
            i += 1
    return i

from collections import *

def removeKthBST(root, k):
    
    dq = deque() #k mem size
    current = root
    count = 0 #for k size
    check = 0
    
    while current or not dq.isEmpty():
        while current:
            dq.appendRight(current)
            count += 1
            if count > k:
                count -= 1
                dq.popLeft()
            current = current.right
        current = dq.popRight()
        count -= 1
        check += 1
        if check == k:
            return current
        current = current.left        
    return None


'''

1
    2
        3
    4        5
c = 3
count = 1
d = 2
check = 1



'''

'''


-1|1,2,3,4,5,|0
0 |1,2,3,4,5|6

s = [0,1,2,3,4,5]
h = 5
w = 6-5 = 1 , ans 4

-1,4,2,5,7,0
0, 1,2,3,4,5

stack = [-1,2,3,4]
i = 5
while 0 < 7:
    h = 7
    #s = [-1,2,5]
    w = 5 - 1 - 4 = 0
'''

def numDecodings(s):
    """
    :type s: str
    :rtype: int
    123454321
    """
    if not s:
        return 0
    memo = [0]*(len(s)+1)
    memo[-1] = 1
    memo[-2] = 0 if s[-1] == '0' else 1
    for i in range(len(s)-2, -1, -1):
        if s[i] == '0':
            continue
        memo[i] = memo[i+1] + memo[i+2] if s[i:i+2] <= '26' else memo[i+1]
    return memo[0]


# print(numDecodings('1238311035781021'))
def isValidBST(root):
    if not root:
        return True
    current = root
    s = []
    last_inorder = None
    while current or len(s) != 0:
        while current:
            s.append(current)
            current = current.left
        current = s.pop()
        if last_inorder:
            if last_inorder.val >= current.val:
                return False
        last_inorder = current
        current = current.right
    return True
    
def connect(root):
    if not root:
        return None
    levels = [root]
    
    while True:
        dummyNode = TreeLinkNode(0)
        current = dummyNode
        
        parentLink = levels[-1]
        while parentLink:
            if parentLink.left:
                current.next = parentLink.left
                current = current.next
            if parentLink.right:
                current.next = parentLink.right
                current = current.next    
            parentLink = parentLink.next
        
        if not dummyNode.next:
            break
        levels.append(dummyNode.next)
        dummyNode.next = None
        
def buy_one_sell_one(prices):
    max_profit = 0
    if not prices:
        return 0
    min_buy_in = prices[0]
    
    for i in range(len(prices)):
        max_profit = max(max_profit,prices[i]-min_buy_in)
        min_buy_in = min(min_buy_in, prices[i])
    return max_profit

def isPalindrome(s):
    if not s:
        return True
    low = 0
    high = len(s)-1
    
    while low < high:
        if not s[low].isalnum():
            low += 1
        elif not s[high].isalnum():
            high -= 1
        else:
            if s[low].lower() != s[high].lower():
                return False
            low += 1
            high -= 1
    return True

def wordBreak(s, words):
    d = [False] * len(s)    
    for i in range(len(s)):
        for w in words:
            if w == s[i-len(w)+1:i+1] and (d[i-len(w)] or i-len(w) == -1):
                d[i] = True
    return d[-1]    


'''
'' a b c
a  0 1 1
b  
c


'''

def edit_distance(word1, word2):
    
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    row = len(word1) + 1
    col = len(word2) + 1
    lastRow = [i for i in range(col)]
    currentRow = lastRow[:]
    print(lastRow)
    for i in range(1,row):
        for j in range(col):
            if not j:
                currentRow[j] = i
                continue
            if word2[j-1] == word1[i-1]:
                currentRow[j] = min(lastRow[j-1],lastRow[j]+1,currentRow[j-1]+1)
            else:
                currentRow[j] = min(lastRow[j-1]+1,lastRow[j]+1,currentRow[j-1]+1)
        lastRow = currentRow[:]
        print(lastRow)
    return currentRow[-1]

# print(edit_distance('', ''))

def minDistance(word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    if not word1:
        return len(word2)
    if not word2:
        return len(word1)
    if not word2 and not word1:
        return 0
    row = len(word1) + 1
    col = len(word2) + 1
    matrix = [[0]*col for i in range(row)]
    for i in range(1,col):
        matrix[0][i] = i
    for i in range(1,row):
        matrix[i][0] = i
    for i in range(1,row):
        for j in range(1,col):
            if word2[j-1] == word1[i-1]:
                matrix[i][j] = min(matrix[i-1][j-1],matrix[i-1][j]+1,matrix[i][j-1]+1)
            else:
                matrix[i][j] = min(matrix[i-1][j-1]+1,matrix[i-1][j]+1,matrix[i][j-1]+1)
    pprint(matrix)
    return matrix[-1][-1]

# print(minDistance('hello','howdy'))


def excel_converted(n):
    string_dic = [char for char in string.ascii_uppercase]
    output = []
    while n:
        output.append(string_dic[(n-1)%26])
        n = (n-1)//26
    return ''.join(output[::-1])

def reverse_llist(head):
    current = head
    prev = None
    while current:
        _next = current.next
        current.next = prev
        prev = current
        current = _next        
    return prev

#implement trie:
class Trie(object):
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = {}
        self.isWord = False
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        root = self
        for char in word:
            if char not in root.children:
                root.children[char] = Trie()
            root = root.children[char]
        # n -> a->p->p_>l->e
        root.isWord = True
        
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        root = self
        for char in word:
            if char not in root.children:
                return False
            root = root.children[char]
        return root.isWord     

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        root = self
        for char in prefix:
            if char not in root.children:
                return False
            root = root.children[char]
        return True  
    
def minSubArrayLen(s, nums):
    if not nums:
        return 0
    i = j = 0
    min_length = float('inf')
    current_sum = 0
    while j < len(nums):
        if i == j and current_sum >= s:
            return 1
        elif current_sum < s:
            current_sum += nums[j]
            j += 1
        else:
            print(nums[i],nums[j-1],current_sum)
            min_length = min(min_length, j-i)
            current_sum -= nums[i]
            i += 1
    while current_sum >= s:
        print(nums[i],nums[j-1],current_sum)
        min_length = min(min_length, j-i)
        current_sum -= nums[i]
        i += 1
    if min_length == float('inf'):
        return 0
    return min_length 
    
def lowestCommonAncestor(root, node1, node2):
    if not root:
        return None
    if root == node1 or root == node2:
        return root
    left = self.lowestCommonAncestor(root.left,node1,node2)
    right = self.lowestCommonAncestor(root.right,node1,node2)
    if right and left:
        return root
    return right or left


def productExceptSelf(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    forward = [1]*len(nums)
    for i in range(1,len(nums)):
        forward[i] *= forward[i-1]
        forward[i] *= nums[i-1]
    print(forward)
    
    backward = [1]*len(nums)
    for i in range(len(nums)-2,-1,-1):
        backward[i] *= backward[i+1]
        backward[i] *= nums[i+1]
    print(backward)
    
    output = [backward[i]*forward[i] for i in range(len(nums))]
    return output

from operator import *

def meeting_rooms(meetings):
    
    meetings.sort(key=itemgetter(0))
    print(meetings)
    for i in range(1,len(meetings)):
        if meetings[i][0] < meetings[i-1][1]:
            return False
    return True
# 
# print(meeting_rooms([[5, 10],[0, 30],[15, 20]])) 

def minMeetingRooms(intervals):
    start = []
    end = []
    starts = [x[0] for x in intervals]
    ends = [x[1] for x in intervals]
    starts.sort()
    ends.sort()
    numRooms = available = s = e = 0
    
    while s < len(starts):
        print('rooms', numRooms)
        print(starts[s], ends[e])
        if starts[s] < ends[e]:
            if not available:
                numRooms += 1
            else:
                available += 1
            s += 1
        else:
            available -= 1
            e += 1

    return numRooms
# 
# meetings = [[5, 10],[0, 30],[15, 20],[20,40],[3,30],[0,50]]
# print(minMeetingRooms(meetings))

def binaryTreePaths(root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
        return []
    paths = [str(root.val)]
    s = [root]
    result = []
    
    while len(s):
        
        node = s.pop()
        current_path = paths.pop()

        if node.left:
            s.append(node.left)
            paths.append(current_path+'->'+str(node.left.val))
            
        if node.right:
            s.append(node.right)
            paths.append(current_path+'->'+str(node.right.val))

        if not node.left and not node.right:
            result.append(current_path)
            
    return result


def h_index(citations):
    n = len(citations)
    buckets = [0]*(n+1)
    for c in citations:
        if c >= n:
            buckets[n] += 1
        else:
            buckets[c] += 1
    count = 0
    for i in range(n,-1,-1):
        count += buckets[i]
        if count >= i:
            return i
    return 0


def addOperators(num, target):
    """
    :type num: str
    :type target: int
    :rtype: List[str]
    """
    if not num:
        return []
    results = []
    for i in range(len(num)):
        helper2(results, num, target, int(num[0:i+1]), [num[0:i+1]], i, int(num[0:i+1])) 
    return results
    
    
def helper2(results, num, target, num_path, string_path, index, last):
    print(num_path, ''.join(string_path), index)
    if num_path == target and index == len(num)-1:
        results.append(''.join(string_path))
    else:
        print(index, len(num))
        for i in range(index+1,len(num)):
            number = num[index+1:i+1]
            helper2(results, num, target, num_path+int(number), string_path+['+',number], i, int(number))
            helper2(results, num, target, num_path-int(number), string_path+['-',number], i, -1*int(number))
            helper2(results, num, target, num_path-last+last*int(number), string_path+['*',number], i, int(number)*last)
            
# print(addOperators('232', 8))
            
            
def insertZeroes(numbers):
    insertPosition = 0
    for num in numbers:
        if num:
            numbers[insertPosition] = num
            insertPosition += 1
    for i in range(insertPosition, len(numbers)):
        numbers[i] = 0
    
    return numbers           


def serialize(self, root):
    """Encodes a tree to a single string.
    
    :type root: TreeNode
    :rtype: str
    """
    data = []
    def helper(node):
        if node:
            data.append(str(node.val))
            helper(node.left)
            helper(node.right)
        else:
            data.append('#')
    helper(root)
    return ' '.join(data)
    
def deserialize(self, data):
    """Decodes your encoded data to tree.
    
    :type data: str
    :rtype: TreeNode
    """
    data = iter(data.split())
    def helper():
        value = next(data)
        if value == '#':
            return None
        node = TreeNode(value)
        node.left = helper()
        node.right = helper()
        return node
    return helper()

def remove_invalid_parentheses(s):
    def isValid(s):
        check = 0
        for char in s:
            if char == '(':
                check += 1
            if char == ')':
                if not check:
                    return False
                check -= 1
        return check == 0
    level = set([s])
    while True:
        run = list(filter(isValid, level))
        if run:
            return run
        level = set([s[:i] + s[i+1:] for s in level for i in range(len(s))])
        
# print(remove_invalid_parentheses('()())()'))


def longest_increasing_path(matrix):
    #elided checks
    row = len(matrix)
    col = len(matrix[0])
    memo = [[0]*col for _ in range(row)]
    longest_path = 0
    for i in range(row):
        for j in range(col):
            longest_path = max(longest_path, dfs(i,j,matrix,memo))
    return longest_path

def dfs(i,j,matrix,memo):
    row = len(matrix)
    col = len(matrix[0])
    if memo[i][j]:
        return memo[i][j]
    directions =[(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
    for change in directions:
        if change[0] >= 0 and change[1] >= 0 and change[0] < row and change[1] < col:
            if matrix[change[0]][change[1]] > matrix[i][j]:
                memo[i][j] = max(memo[i][j], dfs(change[0],change[1],matrix,memo))
    return memo[i][j]+1

    
def maxSubArrayLen(nums, k):
    ans, acc = 0, 0               # answer and the accumulative value of nums
    mp = {0:-1}                 #key is acc value, and value is the index
    for i in range(len(nums)):
        acc += nums[i]
        if acc not in mp:
            mp[acc] = i 
        if acc-k in mp:
            ans = max(ans, i-mp[acc-k])
        pprint(mp)
    return ans


# print(maxSubArrayLen([10,40,20,60,1,  5, -3], 3))




def int_to_english(number):
    ones = {1:'one',9:'nine'}
    teens = {10:'ten', 19:'nineteen'}
    tens = {20:'twenty', 90:'ninety'}
    
    if number < 10:
        return ones[number]
    if number < 20:
        return ones[number%10]
    if number < 100:
        return tens[number%10] + ones[(number//10)%10]
    
    
# matrix = [[1,2],[4,3]]
# print(longest_increasing_path(matrix))