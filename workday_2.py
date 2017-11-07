from pprint import pprint

#Given two strings, find if first string is a subsequence of second

def subOfSecond(string1, string2):
    if len(string2) > len(string1):
        return False
    i = 0
    j = 0
    while i < len(string1) and j < len(string2):
        if string1[i] == string2[j]:
            j += 1
        else:
            j = 0
        i += 1
    return j == len(string2)

def subSeqofFirst(string1, string2):
    i = 0
    j = 0
    while i < len(string1) and j < len(string2):
        if string1[i] == string2[j]:
            j += 1
        i += 1
    return j == len(string2)

def reverse_sentence(sentence):
    '''
    reverse all letters in sentence
    then go through sentence, when you see space, reverse begin and end
    '''
    sentence = list(sentence)
    reverse_string(sentence,0,len(sentence)-1)

    begin = 0

    for i in range(len(sentence)):
        if sentence[i] == ' ':
            reverse_string(sentence,begin,i-1)
            begin = i + 1

    if begin < len(sentence):
        reverse_string(sentence,begin,len(sentence)-1)

    return ''.join(sentence)


def reverse_string(string,low,high):
    while low < high:
        string[low], string[high] = string[high], string[low]
        low += 1
        high -= 1

sentence = 'Hi my name is Petros Dawit. How are you doing?'

# print(reverse_sentence(sentence))

def lonely_island(matrix):
    row = len(matrix)
    col = len(matrix[0])
    visited = set()
    count = 0
    for i in range(row):
        for j in range(col):
            if (i,j) not in visited and matrix[i][j] == 1:
                count += 1
                recurse_find(i,j,row,col,matrix,visited)
    return count

def recurse_find(i,j,row,col,matrix,visited):
    if i < row and j < col and i >= 0 and j >= 0:
        if (i,j) not in visited and matrix[i][j] == 1:
            visited.add((i,j))
            recurse_find(i,j-1,row,col,matrix,visited)
            recurse_find(i,j+1,row,col,matrix,visited)
            recurse_find(i-1,j,row,col,matrix,visited)
            recurse_find(i+1,j,row,col,matrix,visited)
            recurse_find(i-1,j-1,row,col,matrix,visited)
            recurse_find(i+1,j+1,row,col,matrix,visited)
            recurse_find(i-1,j+1,row,col,matrix,visited)
            recurse_find(i+1,j-1,row,col,matrix,visited)


matrix = [[1, 1, 0, 0, 0],[0, 1, 0, 0, 1],[1, 0, 0, 1, 1],
[0, 0, 0, 0, 0],[1, 0, 1, 0, 0]]
# pprint(matrix)
# print(lonely_island(matrix))



'''
original
[1,5,12,15]
[6,7,10,12]
[15,19,22]
[1,5,9,10]

[15]
[10,12]
[15,19,22]
[]

heap = [10,10,12,15]
current_max = 15
min_range = abs(current_max - heap.min) = 4


'''
def rreverse(s):
    def rreverse_helper(s,index):
        if index == len(s)-1:
            return s[index]
        else:
            return rreverse_helper(s,index+1) + s[index]
    return rreverse_helper(s,0)


'''
CHAPTER 7 OOP QUESTIONS CTCI

2.) call centers with 3 employees, f, tl, pm, only 1 tl or pm.

queue system for calls:
    queue for available fresher,
    queue for TL or PM
    respond can't handle it
    employee superclass, with rank passed in constructor

3.) musical jukebox with OOP,

7.5.) online book reader system
user
membership
book
checkout system

7.6) chat server
messages
    message_id pk
    user_id fk
    room_id fk
    description
    media
rooms
    room_id
    users_joined
users
    user_id
    profile_info
    friends(users)





'''
