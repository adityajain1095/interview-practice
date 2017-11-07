from pprint import pprint
'''
String medium hackerrank
'''
#sherlock_holmes and valid number
def isValid(s):
    # Complete this function
    s_dic = {}
    for char in s:
        s_dic[char] = s_dic.get(char,0)
        s_dic[char] += 1
    frequency = -1
    firstChange = False
    for i, char in enumerate(s_dic):
        if i == 0:
            frequency = s_dic[char]
        else:
            if s_dic[char] == frequency:
                continue
            if not firstChange:
                if s_dic[char] - 1 == frequency or s_dic[char] + 1 == frequency or s_dic[char] == 1:
                    firstChange = True
                else:
                    return 'NO'
            else:
                return 'NO'
    return 'YES'

#sherlock holmes and anagrams:
def sherlockAndAnagrams(string):
    # Complete this function
    buckets = {}
    for i in range(len(string)):
        for j in range(1, len(string) - i + 1):
            key = frozenset(Counter(string[i:i+j]).items()) # O(N) time key extract
            buckets[key] = buckets.get(key, 0) + 1
    count = 0
    for key in buckets:
        count += buckets[key] * (buckets[key]-1) // 2
    return count

# q = int(input().strip())
# for a0 in range(q):
#     s = input().strip()
#     result = sherlockAndAnagrams(s)
#     print(result)
import string
import random
s = ''
for i in range(81):
    s += random.choice(string.ascii_uppercase)
# print(s)
# print(len('XECEROUHTIXRYBVNTMWAVSNLPSGSUPVRWHQHIHKRRZORMMUUBTJIVFSCHDDAWHDADAPGHPRAXLIOUS9JO'))

def num_of_paths_to_dest(n):
    if not n:
        return 0
    if n <= 2:
        return 1
    matrix = [[0]*n for i in range(n)]
    for i in range(n):
        matrix[i][0] = 1
    k = 2
    for i in range(n-2,-1,-1):
        for j in range(1,k):
            matrix[i][j] += matrix[i+1][j] + matrix[i][j-1]
        k+=1
    return matrix[0][-2]

def num_of_paths_optimized_space(n):
    #nxn grid
    lastRow = [0]*n
    for i in range(n):
        lastRow[i] = 1
    currentRow = [0]*n
    for i in range(1,n):
        for j in range(n):
            if j == 0:
                currentRow[j] = lastRow[j]
            else:
                currentRow[j] = currentRow[j-1] + lastRow[j]
        lastRow = currentRow
    return currentRow[-1]

print(num_of_paths_optimized_space(3))

#Lily's Homework, find minimal swaps needed to sort in descending or ascending
#just split into two
def solution(a,tag):

    m = {}
    for i in range(len(a)):
        m[a[i]] = i

    sorted_a = sorted(a)
    if tag == 'desc':
        sorted_a = list(reversed(sorted_a))
    ret = 0
    for i in range(len(a)):
        if a[i] != sorted_a[i]:
            ret +=1
            ind_to_swap = m[sorted_a[i]]
            m[a[i]] = m[sorted_a[i]]
            a[i],a[ind_to_swap] = sorted_a[i],a[i]
    return ret

# input()
# a = [int(i) for i in input().split(' ')]
#
# asc=solution(a,'asc')
# desc=solution(a,'desc')
# print(min(asc,desc))


def num_of_paths_to_dest(n):
    if not n:
        return 0
    if n <= 2:
        return 1
    matrix = [[0]*n for i in range(n)]
    for i in range(n):
        matrix[i][0] = 1
    k = 2
    for i in range(n-2,-1,-1):
        for j in range(1,k):
            matrix[i][j] += matrix[i+1][j] + matrix[i][j-1]
        k+=1
    return matrix[0][-2]

print(num_of_paths_to_dest(7))
