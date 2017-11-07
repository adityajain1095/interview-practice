from pprint import pprint
import random
import string
from queue import *
from collections import *


def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """
    current_set = set()
    ans = i = j = 0
    while i < len(s) and j < len(s):
        if s[j] not in current_set:
            current_set.add(s[j])
            j += 1
            print(j)
            ans = max(ans,j-i)
        else:
            current_set.remove(s[i])
            i += 1
    return ans

def median_of_two_sorted_arrays(num1, num2):
    pass

#these are subsequence
def longest_palindromic_substring_count(string):
    if len(string) == 1:
        return 1
    if len(string) == 2:
        if string[0] == string[1]:
            return 2
        else:
            return 1
    longest_substring = 0
    if string[0] == string[-1]:
        longest_substring += 2
        longest_substring += longest_palindromic_substring_count(string[1:-1])
    else:
        longest_substring = max(longest_palindromic_substring_count(string[1:]),longest_palindromic_substring_count(string[:-1]))
    return longest_substring

def longestPalindromeSeq(string):
    string_arr = longestPalindromeSeqHelp(list(string),0,len(string)-1)
    return ''.join(string_arr)

def longestPalindromeSeqHelp(string,i,j):
    if i == j:
        return string[i]
    if i+1 == j:
        if string[i] == string[j]:
            return string[i]+string[j]
        else:
            return string[i]
    longest_substring = ''
    if string[i] == string[j]:
        longest_substring = string[i] + longestPalindromeSeqHelp(string,i+1,j-1) + string[j]
    else:
        left = longestPalindromeSeqHelp(string,i,j-1)
        right = longestPalindromeSeqHelp(string,i+1,j)
        if len(left) > len(right):
            longest_substring = left
        else:
            longest_substring = right
    return longest_substring
# s = 'babadabasbeeassllsreiddkksj'
# print(longestPalindromeSeq(s))
# print(longestPalindromeSeqHelp(s,0,len(s)-1))

#this is substring:
def longestPalindrome(s):
    start = end = 0
    for i in range(len(s)):
        len1 = expandCenter(s,i,i)
        len2 = expandCenter(s,i,i+1)
        max_len = max(len1,len2)
        if max_len > end-start:
            print('max_len',max_len)
            print('i',i)
            start = i - (max_len-1)//2
            end = i + max_len//2
            print(s[start],start,s)
            print(s[end],end,s)
    return s[start:end+1]

def expandCenter(s,left,right):
    while left >= 0 and right <= len(s)-1 and s[left] == s[right]:
        left -= 1
        right += 1
    return right-left-1

def reverse_int(x):
    reversed = 0
    sign = 1
    if x < 0:
        sign = -1
        x *= -1
    while x:
        reversed = reversed*10 + x % 10
        x = x//10
        if reversed >= float('inf') or reversed <= float('-inf'):
            return 0
    return reversed*sign

def maxArea(height):
    """
    :type height: List[int]
    :rtype: int
    """
    max_area = 0
    if not height:
        return max_area
    i, j = 0, len(height)-1
    while i != j:
        min_height = min(height[i],height[j])
        max_area = max(max_area,min_height*(j-i))
        if min_height == height[i]:
            i += 1
        else:
            j -= 1
    return max_area

def romanToInt(self, s):
    """
    :type s: str
    :rtype: int
    """
    dic_roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    integer = 0
    for i in range(len(s)-1):
        first_value = dic_roman[s[i]]
        second_value = dic_roman[s[i+1]]
        if second_value > first_value:
            integer -= first_value
        else:
            integer += first_value
    integer += dic_roman[s[-1]]
    return integer

class CommonPrefix(object):
    """docstring forCommonPrefix."""
    def __init__(self):
        pass

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ''
        prefix = strs[0]
        i = 1
        while i < len(strs) and prefix:
            next_word = strs[i]
            prefix = self.compare_prefix(prefix,next_word)
            i += 1
        return prefix

    def compare_prefix(self,string1,string2):
        prefix = []
        min_length = min(len(string1),len(string2))
        for i in range(min_length):
            if string1[i] == string2[i]:
                prefix.append(string1[i])
            else:
                break
        return ''.join(prefix)

# cp = CommonPrefix()
# print(cp.longestCommonPrefix(['abc','abb']))

'''
three sum to zero
'''
def threeSum(nums):
    res = []
    nums.sort()
    for i in xrange(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l +=1
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res

'''
remove nth from end
'''
def removeNthFromEnd(self, head, n):
    """
    :type head: ListNode
    :type n: int
    :rtype: ListNode
    """
    fast = slow = head
    i = 0
    while i < n and fast:
        fast = fast.next
        i += 1
    if not fast or i < n:
        return head.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head

'''
multiple parentheses check if valid
'''
def isValid(self, s):
    stack = []
    dict = {"]":"[", "}":"{", ")":"("}
    for char in s:
        if char in dict.values():
            stack.append(char)
        elif char in dict.keys():
            if stack == [] or dict[char] != stack.pop():
                return False
        else:
            return False
    return stack == []

'''
merge two linkedlists
'''
def mergeTwoLists(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    if None in (l1,l2):
        return l1 or l2
    if l1.val < l2.val:
        l1.next = self.mergeTwoLists(l1.next,l2)
        return l1
    else:
        l2.next = self.mergeTwoLists(l1,l2.next)
        return l2

'''
swap pairs in linkedlist
'''
def swapPairs(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if not head:
        return head
    if not head.next:
        return head
    first = head
    second = head.next
    while first and second:
        first.val, second.val = second.val, first.val
        first = second.next
        if first:
            second = first.next
    return head

def removeDuplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return None
    i = 1
    while i < len(nums):
        if nums[i] == nums[i-1]:
            nums.pop(i)
        else:
            i += 1
    return nums

def removeDuplicates(A):
    if not A:
        return 0
    end = len(A)
    read = 1
    write = 1
    while read < end:
        if A[read] != A[read-1]:
            A[write] = A[read]
            write += 1
        read += 1
    return write
nums = [1,1,2,2,2,4,6,6,9,10]

#needle in haystack problem, don't check every char jsut make copy, quicker
#especially with repeated long
def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    if not haystack and not needle:
        return 0
    if not haystack:
        return -1
    if not needle:
        return 0
    for i in range(len(haystack)-len(needle)+1):
        if haystack[i:len(needle)+i] == needle:
            return i
    return -1

'''
generate parenthesis combinations all up to n
'''
def generateParenthesis(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    output = self.generateParenthesisHelper([''], 0, 0, n)
    return output

def generateParenthesisHelper(self, string, l, r, n):
    if l+r == n*2:
        return [''.join(string)]
    output = []
    if l < n:
        output.extend(self.generateParenthesisHelper(string+['('], l+1, r, n))
    if r < l:
        output.extend(self.generateParenthesisHelper(string+[')'], l ,r+1, n))
    return output
'''
[1,2,5,4]
[1,4,5,2]
[1,2,3,4]

i = 3
k = 1

[1,2,4,3]
i = 3
k = 2
#if left > right: swap
[1,5,2,4,3]
#once swap, update next loop
#keep doing this till can't swap
[1,5,4,2,3]

[1,2]
i = 1
k = 1
'''

def max_repeating_char(string):
    if not string:
        return 0
    if len(string) == 1:
        return 1
    maximum = current = 1
    for i in range(1,len(string)):
        if string[i] == string[i-1]:
            current += 1
        else:
            current = 1
        maximum = max(current,maximum)
    return maximum

def next_sequence(nums):
    if len(nums) != 1 and nums:

        swap_index = -1

        for i in range(len(nums)-2,-1,-1):
            if nums[i] < nums[i+1]:
                swap_index = i
                break

        if swap_index == -1:
            nums = nums[::-1]
        else:
            reversed_index = -1
            for i in range(len(nums)-1,-1,-1):
                if nums[swap_index] < nums[i]:
                    nums[swap_index], nums[i] = nums[i], nums[swap_index]
                    reversed_index = i
                    break
            nums = nums[:swap_index+1] + list(reversed(nums[swap_index+1:reversed_index+1])) + nums[reversed_index+1:]
            return nums

def rotated_search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    [1,2,3,4]
    low = 0, high = 3
    mid = 1
    low = 2, high = 3
    mid = 2
    low = 3, high = 3
    """
    if not nums:
        return -1
    low, high = 0, len(nums)-1
    while low < high:
        mid = (low + high) / 2
        if nums[mid] == target:
            return mid
        elif nums[low] <= nums[mid]:
            if nums[mid] > target and target >= nums[low]:
                high = mid-1
            else:
                low = mid + 1
        else:
            if nums[high] >= target and target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
    if nums[low] == target:
        return low
    return -1

def searchRange(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    if not nums:
        return [-1,-1]

    low, high = 0 , len(nums)-1
    rang = [-1,-1]

    while low < high:
        mid = (low+high)//2
        if nums[mid] < target:
            low = mid+1
        else:
            high = mid
    if nums[low] == target:
        rang[0] = low

    high = len(nums)-1
    while low < high:
        mid = (low+high)//2+1
        if nums[mid] > target:
            high = mid-1
        else:
            low = mid
    if nums[high] == target:
        rang[1] = low

    return rang

def count_and_say(n):
    if not n:
        return 0
    number_string = ['1']
    if n == 1:
        return 1

    for i in range(1,n):
        new_number_string = []
        c = 1
        for i in range(1,len(number_string)):
            if number_string[i] == number_string[i-1]:
                c += 1
            else:
                new_number_string.append(str(c))
                new_number_string.append(number_string[i-1])
                c = 1
        new_number_string.append(str(c))
        new_number_string.append(number_string[-1])
        number_string = new_number_string
    return int(''.join(number_string))
#
# print(count_and_say(30))

def multiple_strings(num1, num2):
    '''
    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] pos = new int[m + n];

        for(int i = m - 1; i >= 0; i--) {
            for(int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i + j, p2 = i + j + 1;
                int sum = mul + pos[p2];

                pos[p1] += sum / 10;
                pos[p2] = (sum) % 10;
            }
        }

        StringBuilder sb = new StringBuilder();
        for(int p : pos) if(!(sb.length() == 0 && p == 0)) sb.append(p);
        return sb.length() == 0 ? "0" : sb.toString();
    }
    '''
    pass

def rotate_image(matrix):
    row = len(matrix)
    col = len(matrix[0])
    row_i = 0
    col_j = 0
    k = 0
    while row_i < row and col_j < col:
        copy = []
        for j in range(col_j,col):
            copy.append(matrix[row_i][j])

        for i in range(row_i,row):
            copy[i-k], matrix[i][col-1] = matrix[i][col-1], copy[i-k]
        col -= 1
        copy = copy[::-1]

        for j in range(col-1,col_j-1,-1):
            copy[j-k], matrix[row-1][j] = matrix[row-1][j], copy[j-k]
        row -= 1

        for i in range(row-1,row_i-1,-1):
            copy[i-k], matrix[i][col_j] = matrix[i][col_j], copy[i-k]
        col_j += 1
        copy = copy[::-1]

        for j in range(col_j,col):
            copy[j-k], matrix[row_i][j] = matrix[row_i][j], copy[j-k]
        row_i += 1
        k += 1

    return matrix

def rotate_any_image(matrix):
    matrix = matrix[::-1]
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix[i])):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix



def groupAnagrams(strs):
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    """
    sorted_dic = dict()
    for word in strs:
        key = ''.join(sorted(word))
        sorted_dic[key] = sorted_dic.get(key,[])
        sorted_dic[key].append(word)
    output = []
    for key in sorted_dic:
        output.append(sorted_dic[key])
    return output

def max_subarray(nums):
    if not nums:
        return 0
    current = nums[0]
    max_sum = nums[0]
    for i in range(1,len(nums)):
        current = max(current+nums[i],nums[i])
        max_sum = max(current,max_sum)
    return max_sum

def spiral_mxn_matrix(matrix):
    if not matrix:
        return []
    row = len(matrix)
    col = len(matrix[0])
    row_i = 0
    col_i = 0
    output = []

    while row_i < row and col_i < col:
        for i in range(col_i,col):
            output.append(matrix[row_i][i])
        row_i += 1

        for i in range(row_i,row):
            output.append(matrix[i][col-1])
        col -= 1

        for i in range(col-1,col_i-1,-1):
            if row_i != row:
                output.append(matrix[row-1][i])
        row -= 1

        for i in range(row-1,row_i-1,-1):
            if col_i != col:
                output.append(matrix[i][col_i])
        col_i += 1

    return output

def canJump(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    i = reach = 0
    while i <= reach and i < len(nums):
        reach = max(i+nums[i], reach)
        i += 1
    return i == len(nums)

# Definition for an interval.
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

def merge(intervals):
    """
    :type intervals: List[Interval]
    :rtype: List[Interval]
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x.start)
    print(intervals)
    lower_bound = intervals[0].start
    upper_bound = intervals[0].end
    output = []

    for i in range(len(intervals)):
        interval = intervals[i]
        if interval.start <= upper_bound:
            upper_bound = max(upper_bound,interval.end)
        else:
            output.append([lower_bound,upper_bound])
            lower_bound, upper_bound = interval.start, interval.end

    output.append([lower_bound,upper_bound])
    return output

def insert(intervals, newInterval):
    """
    :type intervals: List[Interval]
    :type newInterval: Interval
    :rtype: List[Interval]
    all left that dont merge
    all right that dont merge
    """
    left = []
    right = []

    for interval in intervals:
        if newInterval.start > interval.end:
            left.append(interval)
        if interval.start > newInterval.end:
            right.append(interval)
    print(left)
    if left + right != intervals:
        newInterval.start = min(newInterval.start, intervals[len(left)].start)
        newInterval.end = max(newInterval.end, intervals[-len(right)-1].end)
    return left + [Interval(newInterval.start, newInterval.end)] + right
# print(insert([Interval(0,4),Interval(6,10)], Interval(1,5)))

def generateMatrix(n):
    """
    :type n: int
    :rtype: List[List[int]]
    """
    matrix = [[0]*n for i in range(n)]
    row_i = col_i = 0
    row = col = n
    k = 1
    while row_i < row and col_i < col:
        for i in range(col_i,col):
            matrix[row_i][i] = k
            k += 1
        row_i += 1

        for i in range(row_i,row):
            matrix[i][col-1] = k
            k += 1
        col -= 1

        for i in range(col-1,col_i-1,-1):
            if row_i != row:
                matrix[row-1][i] = k
                k += 1
        row -= 1

        for i in range(row-1,row_i-1,-1):
            if col_i != col:
                matrix[i][col_i] = k
                k += 1
        col_i += 1

    return matrix

def rotateRight(head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    ex:
    [1,2,3,4,5]
    2
    """
    tail = head
    length = 1
    if not head:
        return head

    while tail.next:
        tail = tail.next
        length += 1
    tail.next = head
    k %= length
    if k:
        for i in range(length-k):
            print(tail.val)
            tail = tail.next

    newHead = tail.next
    tail.next = None
    return newHead

def uniquePaths(m, n):
    """
    :type m: int
    :type n: int
    :rtype: int
    """
    if m <= 0 or n <= 0:
        return 0
    lastRow = []
    for i in range(n):
        lastRow.append(1)
    currentRow = lastRow[:]
    for i in range(1,m):
        for j in range(1,n):
            currentRow[j] = lastRow[j] + currentRow[j-1]
        lastRow = currentRow[:]
    return currentRow[-1]

#unique Paths 2
import itertools

class Paths(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if m <= 0 and n <= 0:
            return 0
        return self.nCr(m+n-2,n-1)

    def nCr(self,n,r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)

def uniquePathsWithObstacles(obstacleGrid):
    """
    :type obstacleGrid: List[List[int]]
    :rtype: int
    """

    row = len(obstacleGrid)
    col = len(obstacleGrid[0])
    lastRow = [0]*col

    for i in range(col):
        if obstacleGrid[0][i] == 1:
            break
        else:
            lastRow[i] = 1

    currentRow = lastRow[:]
    for i in range(1,row):
        for j in range(col):
            if obstacleGrid[i][j] == 1:
                currentRow[j] = 0
            elif j == 0:
                currentRow[j] = lastRow[j]
            else:
                currentRow[j] = currentRow[j-1] + lastRow[j]
        lastRow = currentRow[:]

    return currentRow[-1]

def minPathSum(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if not grid:
        return 0

    row = len(grid)
    col = len(grid[0])

    lastRow = [grid[0][0]]
    for i in range(1,col):
        lastRow.append(lastRow[i-1]+grid[0][i])

    currentRow = lastRow[:]
    for i in range(1,row):
        for j in range(col):
            if j == 0:
                currentRow[j] = lastRow[j] + grid[i][j]
            currentRow[j] = grid[i][j]
            currentRow[j] += min(currentRow[j-1],lastRow[j])
        lastRow = currentRow[:]

    return currentRow[-1]


def climbStairs(n):
    """
    :type n: int
    :rtype: int
    screams recursion
    """
    if n <= 0:
        return
    if n == 1:
        return 1
    if n == 2:
        return 2
    x = 2
    y = 1
    z = 0
    for i in range(2,n):
        z = x + y
        y = x
        x = z
    return z

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


def minPathSum(grid):
    if not grid:
        return 0
    row = len(grid)
    col = len(grid[0])
    lastRow = []
    for x in grid[0]:
        lastRow.append(x)
    currentRow = lastRow[:]
    for i in range(1,row):
        for j in range(col):
            if j == 0:
                currentRow[j] = lastRow[j]
            else:
                currentRow[j] = currentRow[j-1] + lastRow[j]
        lastRow = currentRow[:]
    return currentRow[-1]

matrix = [[1,2,3,4]]

def searchMatrix(matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    if not matrix:
        return False
    row = len(matrix)
    col = len(matrix[0])
    i = j = 0
    while i < row and j < col:
        if matrix[i][j] > target:
            return False
        elif matrix[i][j] == target:
            return True
        elif j == col-1:
            if target > matrix[i][j]:
                j = 0
            i += 1
        elif i < row-1:
            if target >= matrix[i+1][j]:
                i += 1
            else:
                j += 1
        else:
            j += 1
    return False

'''
want nums = [0,0,0,0,1,1,1,1,1,2,2,2,2,2]
in place
'''

def sortColors(nums):
    if not nums:
        return []
    j = 0
    k = len(nums)-1
    for i in range(k+1):
        while nums[i] == 2 and i < k:
            nums[k], nums[i] = nums[i], nums[k]
            k -= 1
        while nums[i] == 0 and i > j:
            nums[j], nums[i] = nums[i], nums[j]
            j += 1
    return nums

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
    if row == 1 and col == 1 and len(word) == 1:
        if board[0][0] == word[0]:
            return True

    for i in range(row):
        for j in range(col):
            if self.findWord(i,j,row,col,board,word,0):
                return True
    return False

def findWord(i,j,row,col,board,word,k):
    if i < row and j < col and i >= 0 and j >= 0:
        if k == len(word):
            return True
        else:
            if board[i][j] == word[k]:
                w = self.findWord(i,j-1,row,col,board,word,k+1)
                x = self.findWord(i,j+1,row,col,board,word,k+1)
                y = self.findWord(i+1,j,row,col,board,word,k+1)
                z = self.findWord(i-1,j,row,col,board,word,k+1)
                return w | x | y | z
    return False


def sorted_remove_dup(nums):

    if len(nums) < 2:
        return len(nums)
    i = 2
    while i < len(nums):
        if nums[i] == nums[i-1] and nums[i] == nums[i-2]:
            nums.pop(i)
        else:
            i += 1
    print(nums)
    return len(nums)

def sorted_arr_find(nums, target):

    if not nums:
        return False

    low = 0
    high = len(nums)-1

    while low <= high:
        mid = (low+high)//2
        if nums[mid] == target:
            return True
        if nums[low] == nums[mid] and nums[mid] == nums[high]:
            low += 1
            high -= 1
        elif nums[low] <= nums[mid]:
            if nums[mid] > target and target >= nums[low]:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if nums[high] >= target and target > nums[mid]:
                low = mid + 1
            else:
                high = mid -1

    return False

target = 1
'''
[1,4,5,3,2,7,8,1,5,6,7,8,9]
prev = None
head = 4
switch = 1
first case:
    make 1 head, head 1. next  = 4
else:
    make head.next = switch_node

prev.next = switch_node.next
'''

[1,4,5,3,2,7,8,5,6,7,8,9]

class LinkNode(object):

    def __init__(self, data):
        self.val = data
        self.next = None

head = LinkNode(4)
a = LinkNode(5)
b = LinkNode(3)
c = LinkNode(2)
d = LinkNode(7)
e = LinkNode(8)
f = LinkNode(1)
g = LinkNode(5)
h = LinkNode(6)
i = LinkNode(7)
j = LinkNode(8)
k = LinkNode(9)
head.next = a
a.next = b
b.next = c
c.next = d
d.next = e
e.next = f
f.next = g
g.next = h
h.next = i
i.next = j
j.next = k

def partition(head, x):
    """
    :type head: ListNode
    :type x: int
    :rtype: ListNode
    """

    h1 = l1 = ListNode(0)
    h2 = l2 = ListNode(0)
    while head:
        if head.val < x:
            l1.next = head
            l1 = l1.next
        else:
            l2.next = head
            l2 = l2.next
        head = head.next
    l2.next = None
    l1.next = h2.next
    return h1.next

'''

[1,4,5,3,2,7,8,1,5,6,7,8,9]

h1 = l1 = ListNode(0)
h2 = l2 = ListNode(0)
while head:
    if head.val < x:
        l1.next = head
        l1 = l1.next
    else:
        l2.next = head
        l2 = l2.next
    head = head.next
l2.next = None
l1.next = h2.next
return h1.next

h1 = l1 = 0
h2 = l2 = 0

1    l1.next = 1
    l1 = 1

2     l2.next = 4
        l2 = 4

0->1->1->4
0->4->5->3->2->7->8->5-6-7-8-9->none
l1.next = 4
start at h1.next

'''

def merge_enough_space(nums1, m, nums2, n):
    k = m+n-1
    j = n-1
    i = m-1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1

    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1

    return nums1

# def reverseBetween(head, m, n):
#         """
#     :type head: ListNode
#     :type m: int
#     :type n: int
#     :rtype: ListNode
#     """
#     dummy = ListNode(0)
#     dummy.next = head
#     p = dummy
#
#     for i in range(1,m):
#         p = p.next
#
#     prev = None
#     current = p.next
#     for i in range(m,n+1):
#         next_temp = current.next
#         current.next = prev
#         prev = current
#         current = next_temp
#
#     p.next.next = current
#     p.next = prev
#
#     return dummy.next

#[1,2,3,    ,8,7,6,5      ,9,10,12     ]
#     p      prev         current
#iterative inorder reversal
def inorderReversal(root):
    s = []
    output = []
    current = root
    while len(s) != 0 or current:
        while current:
            s.append(current)
            current = current.left
        current = s.pop()
        output.append(current.val)
        current = current.right

    return output



def isSameTree(p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    if not p and not q:
        return True
    if not p and q:
        return False
    if not q and p:
        return False
    if p.val != q.val:
        return False
    left = self.isSameTree(p.left,q.left)
    right = self.isSameTree(p.right, q.right)
    return left and right


class Node(object):

    def __init__(self, data=None):
        self.val = data
        self.right = None
        self.left = None

def isSymmetric(root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    #check if left is mirror of right
    if not root:
        return True
    return isSymmetricMirror(root.left, root.right)

def isSymmetricMirror(left, right):
    if not left and not right:
        print('1')
        return True
    if not left and right:
        print('2')
        return False
    if not right and left:
        print('3')
        return False
    if left.val != right.val:
        print('4')
        return False
    return isSymmetricMirror(left.right,right.left) and isSymmetricMirror(left.left, right.right)

arr = [1,2,2,3,4,4,3]



root = Node(1)
root.left = Node(2)
root.left.left = Node(3)
root.left.right = Node(4)
root.right = Node(2)
root.right.left = Node(4)
root.right.right = Node(3)



def levelOrder(root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """

    if not root:
        return []

    nodes = [[root]]
    depth = [[root.val]]

    while nodes[-1]:
        leaves = []
        depth_values = []
        for node in nodes[-1]:
            if node.left:
                leaves.append(node.left)
                depth_values.append(node.left.val)
            if node.right:
                leaves.append(node.right)
                depth_values.append(node.right.val)
        nodes.append(leaves)
        depth.append(depth_values)

    depth.pop()
    return depth

def buildTree(preorder, inorder):
    if inorder:
        ind = inorder.index(preorder.pop(0))
        root = TreeNode(inorder[ind])
        root.left = self.buildTree(preorder, inorder[0:ind])
        root.right = self.buildTree(preorder, inorder[ind+1:])
        return root
    return None

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
    root.left = self.sortedArrayHelper(nums,low,mid-1)
    root.right = self.sortedArrayHelper(nums,mid+1,high)
    return root

def maxDepth(root,depth=0):
    if not root:
        return depth

    left = maxDepth(root.left,depth+1)
    right = maxDepth(root.right,depth+1)

    return max(left,right)

# print(maxDepth(root,0))

def isBalanced(root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    return self.isBalancedHelper(root) != -1

def isBalancedHelper(root):
    if not root:
        return 0
    left = isBalancedHelper(root.left)
    right = isBalancedHelper(root.right)
    if left == -1 or right == -1 or abs(left-right) > 1:
        return -1
    return 1 + max(left,right)

def hasPathSum(root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if not root:
        return False
    if root.val == sum and not root.right and not root.left:
        return True
    left = hasPathSum(root.left,sum-root.val)
    right = hasPathSum(root.right,sum-root.val)
    return left or right

def pascal(numRows):
    if numRows == 0:
        return []
    if numRows == 1:
        return [[1]]
    if numRows == 2:
        return [[1],[1,1]]
    output = [[1],[1,1]]

    for i in range(3,numRows+1):
        row = [1]
        for i in range(0,len(output[-1])-1):
            row.append(output[-1][i]+output[-1][i+1])
        row.append(1)
        output.append(row)

    return output

def max_profit(prices):
    if not prices:
        return 0
    if len(prices) == 1:
        return 0
    maximum = 0
    cheapest_buy = prices[0]
    for i in range(1,len(prices)):
        maximum = max(prices[i]-cheapest_buy,maximum)
        cheapest_buy = min(prices[i],cheapest_buy)
    return maximum

def max_profit(prices):
    #only buy or sell on 1 day, can't buy and sell on both days
    maximum = 0
    bought = prices[0]
    i = 1
    while i < len(prices)-1:
        if prices[i] - bought > 0 and prices[i] > prices[i+1]:
            maximum += (prices[i] - bought)
            bought = prices[i+1]
            i += 2
        else:
            bought = min(bought,prices[i])
            i += 1
    if prices[-1] > prices[-2]:
        maximum += (prices[-1] - bought)
    return maximum

#
# prices = [7, 1, 5, 3, 4, 6]
#
# print(max_profit(prices))

def valid_palindrome(string):
    low = 0
    high = len(string)-1
    while low < high:
        if string[low] != string[high]:
            return False
        low -= 1
        high += 1
    return True

def word_ladder(beginWord, endWord, wordList):
    s = []
    count = 1
    wordListSet = set(wordList)
    visited = set()
    s.append([beginWord])
    while len(s) != 0:
        breadth = s.pop()
        print(breadth)
        new_breadth = []
        for i in range(len(breadth)):
            word = breadth[i]
            if word == endWord:
                return count
            new_breadth.extend(get_neighbours(word, visited, wordListSet))
        if new_breadth:
            count += 1
            s.append(new_breadth)
    return 0

def get_neighbours(word, visited, wordListSet):
    neighbors = []
    for i in range(len(word)):
        for char in string.ascii_lowercase:
            new_word = word[:i]+char+word[i+1:]
            if new_word not in visited and new_word in wordListSet:
                neighbors.append(new_word)
                visited.add(new_word)
    return neighbors

# word_list = ["hot","dot","dog","lot","log","cog"]
#
# print(word_ladder('hit','cog',word_list))

def longest_consecutive_sequence(arr):
    #dont sort
    counts = set(arr)
    maximum = 0
    for x in arr:
        if x-1 not in counts:
            y = x+1
            while y in counts:
                y += 1
            maximum = max(y-x, maximum)
    return maximum

# print(longest_consecutive_sequence([100, 4, 200, 1, 3, 2]))

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

def root_to_leaf_sum(root, sum):
    if not root.left and not root.right:
        return sum
    left = right = 0
    if root.left:
        left = root_to_leaf_sum(root.left, sum*10+root.left.val)
    if root.right:
        right = root_to_leaf_sum(root.right, sum*10+root.right.val)
    return left+right

#
# print(root_to_leaf_sum(root, root.val))

def surrounded_x(matrix):

    row = len(matrix)
    col = len(matrix[0])
    visited = set()

    for i in range(row):
        if not matrix[i][0] and matrix[i][0] not in visited:
            find_neighbors(i, 0, visited, row, col, matrix)
        if not matrix[i][col-1] and matrix[i][col-1] not in visited:
            find_neighbors(i, col-1, visited, row, col, matrix)

    for j in range(col):
        if not matrix[0][j] and matrix[0][j] not in visited:
            find_neighbors(0, j, visited, row, col, matrix)
        if not matrix[row-1][j] and matrix[row-1][j] not in visited:
            find_neighbors(row-1, j, visited, row, col, matrix)

    for i in range(row):
        for j in range(col):
            if not matrix[i][j]:
                matrix[i][j] = 2

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 1:
                matrix[i][j] = 0

    return matrix


def find_neighbors(i, j, visited, row, col, matrix):
    s = []
    s.append((i,j))
    while len(s) != 0:
        node = s.pop()
        index_i = node[0]
        index_j = node[1]
        if (index_i,index_j) not in visited and index_i >= 0 and index_j >= 0 and index_i < row and index_j < col:
            if matrix[index_i][index_j] == 0:
                visited.add((index_i,index_j))
                matrix[index_i][index_j] = 1
                s.append((index_i-1, index_j))
                s.append((index_i+1, index_j))
                s.append((index_i, index_j-1))
                s.append((index_i, index_j+1))


# matrix = [[2,2,2,2],[2,0,0,2],[2,2,0,2],[2,0,2,2]]
# print(surrounded_x(matrix))


def partition_palindrome(palindrome):
    # output = []
    # output.append(list(palindrome))
    # s = []
    # s.append(palindrome)
    # if check(palindrome):
    #     output.append(palindrome)
    # while len(s) != 0:
    #     string = s.pop()
    #     for
    pass

def check(string):
    low = 0
    high = len(string)-1
    while low < high:
        if string[low] != high:
            return False
        low += 1
        high -=1
    return True

'''


     a     a     b

a    T    T      F

a         T      F

b                T


2d matrix

if a[i] == b[j] and (abs(i-j) < 2 or matrix[i+1][j-1])

'''

def singleNumber(nums):
    #use xOR, commutative
    result = 0
    for num in nums:
        result ^= num
        print(result)
    return result


def wordBreak(s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    while s:
        t = True
        for word in wordDict:
            if s.startswith(word):
                s = s[len(word):]
                wordDict.remove(word)
                t = False
                break
        if t:
            return False
    if wordDict:
        return False
    return True

s = 'leetcode'
wordDict = list(s)
# print(wordBreak(s, wordDict))

# s = 'code'
# s = s[4:]
# print(s)

def hasCycle(head):
    if not head:
        return False
    first = head
    second = head
    while second and second.next:
        first = first.next
        second = second.next.next
        if first == second:
            return True
    return False


def reorderList(head):
        first, middle = split(head)
        middle = reverse(middle)
        return merge(first,middle,0)

def split(head):
    dummy = ListNode(0)
    middle = head
    dummy.next = middle
    current = head
    index = 0
    while current.next != None:
        if index % 2:
            middle = middle.next
        current = current.next
        index += 1
    result = middle.next
    middle.next = None
    return dummy.next, result


def reverse(head):
    previous = None
    while head:
        next = head.next
        head.next = previous
        previous = head
        head = next
    return previous

def merge(head1,head2,index):
    # dummy = ListNode(0)
    # dummyReturn = dummy
    if not head1 and not head2:
        return None
    if not head1:
        return head2
    if not head2:
        return head1
    if index % 2 == 0:
        head1.next = merge(head1.next,head2,index+1)
        return head1
    else:
        head2.next = merge(head1,head2.next,index+1)
        return head2

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
g = ListNode(7)
h = ListNode(8)
a.next = b
b.next = c
c.next = d
d.next = e
e.next = f
f.next = g
g.next = h

# x = reorderList(a)
# print(x.data)
# print(x.next.data)
# x = x.next.next
# print(x.data)
# print(x.next.data)
# x = x.next.next
# print(x.data)
# print(x.next.data)
# x = x.next.next
# print(x.data)
# print(x.next.data)

def preorderTraversal(root):
    if not root:
        return []
    s = []
    output = []
    s.append(root)
    while len(s) != 0:
        node = s.pop()
        output.append(node.val)
        if node.right:
            s.append(node.right)
        if node.left:
            s.append(node.left)
    return output

def postorderTraversal(root):
    if not root:
        return []
    s1 = [root]
    s2 = []
    output = []

    while len(s1):
        node = s1.pop()
        s2.append(node)
        if node.left:
            s1.append(node.left)
        if node.right:
            s1.append(node.right)

    while len(s2):
        output.append(s2.pop().val)

    return output

def inorderTraversal(root):
    if not root:
        return []

    s = []
    current = root
    done = 0
    output = []

    while not done:
        if current:
            s.append(current)
            current = current.left
        else:
            if len(s):
                current = s.pop()
                output.append(current.val)
                current = current.right
            else:
                done = 1
    return output

def inorderTraversal2(root):
    if not root:
        return []
    current = root
    s = []
    output = []
    while len(s) != 0 or current:
        while current:
            s.append(current)
            current = current.left
        current = s.pop()
        output.append(current.val)
        current = current.right
    return output

# print(preorderTraversal(root))
# print(inorderTraversal(root))
# print(inorderTraversal2(root))
# print(postorderTraversal(root))

def split(head):
    dummy = ListNode(0)
    fast = head
    slow = head
    while fast and fast.next:
        dummy = slow
        slow = slow.next
        fast = fast.next.next
    dummy.next = None
    return head, slow

# def mergeSort(head1,head2):
#     dummy = ListNode(0)
#     dummyReturn = dummy
#     while head1 and head2:
#         if head1.data < head2.data:
#             dummy.next = head1
#             head1 = head1.next
#         else:
#             dummy.next = head2
#             head2 = head2.next
#         dummy = dummy.next
#     if head1:
#         dummy.next = head1
#     if head2:
#         dummy.next = head2
#     return dummyReturn.next



def mergeSort(head1,head2):
    if not head1 and not head2:
        return None
    if not head1:
        return head2
    if not head2:
        return head1
    if head1.data < head2.data:
        head1.next = mergeSort(head1.next,head2)
        return head1
    else:
        head2.next = mergeSort(head1,head2.next)
        return head2

def sortList(head):
    if not head or not head.next:
        return head

    front, back = split(head)

    return mergeSort(sortList(front),sortList(back))


a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
d = ListNode(4)
e = ListNode(5)
f = ListNode(6)
g = ListNode(7)
h = ListNode(8)
a.next = b
b.next = c
c.next = d
d.next = e
e.next = f
f.next = g
g.next = h

# x = sortList(reorderList(a))
#
#
# print(x.data)
# print(x.next.data)
# print(x.next.next.data)
# print(x.next.next.next.data)

def reverseWords(s):
    s = list(s)
    reverse(s,0,len(s)-1)
    i = begin = 0
    while i < len(s):
        if s[i] == ' ':
            reverse(s,begin,i-1)
            begin = i + 1
        i += 1
    reverse(s,begin,len(s)-1)
    return s


def reverse(s, low, high):
    while low < high:
        s[low], s[high] = s[high], s[low]
        low += 1
        high -= 1
    return s

# s = 'i fucking hate interviews '
# print(reverseWords(s))

def maximum_product_subarray(nums):

    if len(nums) < 1:
        return
    maximum = min_val = max_val = nums[0]

    for i in range(1,len(nums)):
        #always takes into account nums[i], and then max, min behind it
        max_val, min_val = max(nums[i],max(nums[i]*max_val,nums[i]*min_val)), min(nums[i],min(nums[i]*min_val,nums[i]*max_val))
        print(max_val, min_val)
        maximum = max(max_val,maximum)
    return maximum

# print(maximum_product_subarray([1,2,3,-2,4,-4,-40]))




def maximum_product_subsequence(nums):
    negatives = 0
    negativeCheck = False
    min_negative = 0

    for x in nums:
        if x < 0:
            negatives += 1

    if negatives % 2 == 1:
        negativeCheck = True
        min_negative = min(nums)

    if negativeCheck == 1 and len(nums) == 2 and 0 in nums:
        return min_negative

    result = 1
    for x in nums:
        if x != 0:
            if negativeCheck:
                if x == min_negative:
                    negativeCheck = False
                else:
                    result *= x
            else:
                result *= x

    return result

# print(maximum_product_subsequence([0,-3]))

def findMin(nums):
    #rotate sorted array
    low = 0
    high = len(nums)-1
    while low < high:
        if nums[low]<nums[high]:
            return nums[low]
        mid = (low+high)//2
        if nums[mid] >= nums[low]:
            low = mid+1
        else:
            high = mid
    return nums[low]

# print(findMin([1,2,-2,-1,0]))


def intersection_of_linked_lists(head1, head2):
    #you can go to head of other once we reach end, no need for difference
    head1length = 0
    head2length = 0
    head1copy = head1
    head2copy = head2
    while head1copy:
        head1length += 1
        head1copy = head1copy.next
    while head2copy:
        head2length += 1
        head2copy = head2copy.next
    if head1length > head2length:
        while head1length - head2length > 0:
            head1 = head1.next
            head1length -= 1
    else:
        while head2length - head1length > 0:
            head2 = head2.next
            head2length -= 1

    while head1:
        if head1 == head2:
            return head1
        head1 = head1.next
        head2 = head2.next
    return None

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def calculate(expression):
    expression = list(expression)
    s = []
    result = 0
    sign = 1
    for x in expression:
        if isInt(x):
            result += sign*int(x)
        elif x == '-':
            sign = -1
        elif x == '+':
            sign = 1
        elif x == '(':
            s.extend([result, sign])
            result = 0
            sign = 1
        else:
            result += s.pop()*s.pop()
    if len(s) == 0:
        return result
    else:
        raise ValueError('bad input')

# print(calculate('(1+(4+5+2)-3)+(6+8)'))

def combinationSum(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], res)
    return res

def dfs(self, nums, target, index, path, res):
    if target > 0:
        if target == 0:
            res.append(path)
            return
        for i in range(index, len(nums)):
            self.dfs(nums, target-nums[i], i, path+[nums[i]], res)

'''
end
'''
import collections

arr = [[1,3],[2,6],[8,10],[15,18]]

def merge_intervals(ranges):
    if not ranges:
        return None
    current_start = ranges[0][0]
    current_end = ranges[0][1]
    output = []
    for arr in ranges:
        if arr[0] > current_end:
            output.append([current_start,current_end])
            current_start = arr[0]
            current_end = arr[1]
        else:
            current_end = max(current_end, arr[1])
    output.append([current_start, current_end])
    return output
# print(merge_intervals(arr))


arr = [0,10,8,9,6,7,4,10,0]

def longest_distance(seq):
    res, ltr, rtl = 0, 0, 0
    for i in range(len(seq)):
        if seq[i] >= seq[ltr]:
            res = max(res, i-ltr+1)
            ltr = i
        if seq[-i-1] >= seq[-rtl-1]:
            res = max(res, i-rtl+1)
            rtl = i
    return res
        
# print(longest_distance(arr))
    
import string
def excel_sheet(n):
    string_dic = {}
    
    for i, char in enumerate(string.ascii_uppercase,1):
        string_dic[i] = char
    string_dic[0] = 'Z'
    
    output = ''
    m = 0
    while n:
        m = n%26
        output = string_dic[m] + output
        n //= 26
        if not m:
            n -= 1
    return output
        

# output = []
# for i in range(1,1000):
#     output.append(excel_sheet(i))
# pprint(output)


def happy_number(n):
    cycle = {}
    while n and n not in cycle:
        cycle[n] = True
        m = 0
        while n:
            n, o = divmod(n,10)
            o *= o
            m += o
        if m == 1:
            return True
        n = m
    return False

# print(happy_number(18181))


def reverse_llist(head):
    prev = None
    current = head
    
    while current:
        next = current.next
        current.next = prev
        prev = current
        current = next
        
    return prev


class Trie(object):
    
    class TrieNode(object):
        def __init__(self):
            self.childrenNode = {}   
            self.isWord = False

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = self.TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        level = self.root
        for char in word:
            if char not in level.childrenNode:
                level.childrenNode[char] = self.TrieNode()
            level = level.childrenNode[char]
        level.isWord = True  

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        level = self.root
        for char in word:
            if char not in level.childrenNode:
                return False
            level = level.childrenNode[char]
        return level.isWord    

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        level = self.root
        for char in prefix:
            if char not in level.childrenNode:
                return False
            level = level.childrenNode[char]
        return True  
        
# t = Trie()
# t.insert('hello')
# # print(t.root.childrenNode)
# # print(t.root.childrenNode['h'].childrenValues)
# t.insert('hi')        
# # print(t.root.childrenNode)
# # print(t.root.childrenNode['h'].childrenNode['i'].childrenValues)
# print(t.startsWith('hel'))

def search_range(nums, target):
    def search(target):
        low, high = 0, len(nums)-1
        while low <= high:
            mid = (low+high)//2
            if nums[mid] >= target:
                high = mid-1
            else:
                low = mid + 1
        return low    
    low = search(target)
    return [low, search(target+1)-1] if target in nums[low:low+1] else [-1, -1]


# nums = [8]
# print(search_range(nums,8))


def permutations(nums):
    permutations = [[]]
    for num in nums:
        new_permutations = []
        for perm in permutations:
            for i in range(len(perm)+1):
                new_permutations.append(perm[:i]+[num]+perm[i:])
        permutations = new_permutations
    return permutations



    
def isBalanced(root):
    return max_length(root) - min_length(root) <= 1

def max_length(root):
    if not root:
        return 0
    return 1 + max(max_length(root.right),max_length(root.left))

def min_length(root):
    if not root:
        return 0
    return 1 + min(min_length(root.right),min_length(root.left))


def ladderLength(beginWord, endWord, wordList):
    visited = set()
    s = []
    s.append(beginWord)
    visited.add(beginWord)
    distance = 0
    while len(s) != 0:
        word = s.pop()
        for i in range(len(word)):
            for l in string.ascii_lowercase:
                new_word = word[:i] + l + word[i+1:]
                print(new_word)
                if new_word == endWord:
                    return distance + 1
                if new_word not in visited and new_word in wordList:
                    print('hi')
                    s.append(new_word)
                    visited.add(new_word)
    return 0




def excel_sheet(n):
    output = ['']
    string_dic = {}
    i = 0
    for char in string.ascii_lowercase:
        string_dic[i] = char 
        i += 1
    while n:
        m = (n-1)%26
        output = [string_dic[m]]+output
        n = (n-1)//26
    return ''.join(output)





































































