#Given two strings, find if first string is a subsequence of second

def subStringSeqSearch(str1, str2):
    m = len(str1)
    n = len(str2)
    j = 0
    i = 0
    while i < m and j < n:
        if str1[i] == str2[j]:
            j += 1
        i += 1
    return j == n

def subStringSearch(str1,str2):
    m = len(str1)
    n = len(str2)
    j = 0
    i = 0
    while i < m and j < n:
        if str1[i] == str2[j]:
            j += 1
        else:
            j = 0
        i += 1
    return j == n

def reverse_list(letters, first, last):
    "reverses the elements of a list in-place"
    while first < last:
        letters[first], letters[last] = letters[last], letters[first]
        first += 1
        last -= 1

def reverse_words(string):
    """reverses the words in a string using a list, with each character
    as a list element"""
    characters = list(string)
    reverse_list(characters,0,len(characters)-1)
    first = last = 0
    while first < len(characters) and last < len(characters):
        if characters[last] != ' ':
            last += 1
        else:
            reverse_list(characters, first, last-1)
            last += 1
            first = last
    if first < last:
        reverse_list(characters, first, last=len(characters)-1)
    return ''.join(characters)

def island(matrix):
    if not matrix:
        return 0
    row = len(matrix)
    col = len(matrix[0])
    visited = set()
    count = 0
    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 1 and (i,j) not in visited:
                find_island(i,j,row,col,matrix,visited)
                count += 1
    return count

def find_island(i,j,row,col,matrix,visited):
    if 0 <= i and i < row and 0 <= j and j < col:
        if matrix[i][j] == 1 and (i,j) not in visited:
            visited.add((i,j))
            find_island(i-1,j,row,col,matrix,visited)
            find_island(i+1,j,row,col,matrix,visited)
            find_island(i,j-1,row,col,matrix,visited)
            find_island(i,j+1,row,col,matrix,visited)
            find_island(i-1,j-1,row,col,matrix,visited)
            find_island(i-1,j+1,row,col,matrix,visited)
            find_island(i+1,j-1,row,col,matrix,visited)
            find_island(i+1,j+1,row,col,matrix,visited)
matrix = [[1, 1, 0, 0, 0],[0, 1, 0, 0, 1],[1, 0, 0, 1, 1],
[0, 0, 0, 0, 0],[1, 0, 1, 0, 1]]
print(island(matrix))

'''
There are k lists of sorted integers. Make a min heap of size k containing 1 element
from each list.Keep track of min and max element and calculate the range.
In min heap, minimum element is at top. Delete the minimum element and another element
instead of that from the same list to which minimum element belong. Repeat the process
till any one of the k list gets empty. Keep track of minimum range.
'''

def rreverse(s):
    def rreverse_helper(s,index):
        if index == len(s)-1:
            return s[index]
        else:
            return rreverse_helper(s,index+1) + s[index]
    return rreverse_helper(s,0)
print(rreverse('hello'))
# print(reverse_words('getting good at coding needs a lot of practice'))
