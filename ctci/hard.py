from pprint import pprint
from collections import deque
import random

def perfect_shuffle(deck):
    for i in range(len(deck)):
        random_num = random.randrange(i,len(deck))
        deck[i], deck[random_num] = deck[random_num], deck[i]
    return deck

def random_gen_from_arr(arr,m,n):
    output = set()
    clone = arr.copy()
    while len(output) != m and i < n:
        random_num = random.randrange(i,len(clone))
        clone[i], clone[random_num] = clone[random_num] , clone[i]
        output.add(clone[i])
    if len(output) != m:
        return False
    return output

#think of mods instead of checking for each one, break up into each one, ten, hundreds ...
def count_twos_between(n):
    count = 0
    for i in range(1,n):
        twos = 0
        string = str(i)
        for k in range(len(string)):
            if string[k] == '2':
                twos += 1
        count += twos
    return count

def search_distance(word1,word2,text_file):
    min_distance = float('inf')
    word1_position = float('inf')
    word2_position = float('-inf')
    with open('text_file') as f:
        i = 0
        positions = {}
        for line in f:
            for word in line:
                if word == word1:
                    word1_position = i
                if word == word2:
                    word2_position = i
                min_distance = abs(word1_position - word2_position)
                i += 1
    return min_distance

#select median, then choose num greater than that
def select(arr,k):
    n = len(arr)
    if not 0 <= k < n:
        return False
    pivot = random.choice(arr)
    L, E, G = [],[],[]
    for x in arr:
        if x < pivot:
            L.append(x)
        elif x == pivot:
            E.append(x)
        else:
            G.append(x)
    if k < len(L):
        return select(L,k)
    elif k < len(L) + len(E):
        return pivot
    else:
        return select(G, k - len(E) - len(L))

# arr = [1,6,2,0,1,-1,-5,3,20,17,-2,-4,-7,10,12]
# print(arr)
# print(sorted(arr))
# print(select(arr,5))
# print(select(arr,100))

def double_word(arr):
    arr_set = set(arr)
    max_word = ''
    max_word_length = 0
    for word in arr:
        for i in range(len(word)):
            if word[:i] in arr_set and word[i:] in arr_set:
                if max_word_length > len(word):
                    max_word = word
    return word

arr_words = ['test', 'tester', 'testertest', 'testing', 'testingtester']
print(double_word(arr_words))

#two heaps
def new_median_keep_up(arr,number):
    '''
    median always is max heap of min
    greater than median heap is minheap of values
    less than median is maxheap of values

    if length of greater than bigger than smaller than by more than +1,
        pop min value and put in smaller than

    if length of smaller than bigger than greater than by more than +1.
        pop max value and put in bigger than

    return smaller than maxheap.max()

    #O(log(n)) insert each time
    '''
    pass

def change_words(word1, word2):
    s = []
    s.append(word1)
    word1.visited = True
    while len(s) != 0:
        word = s.pop()
        for close_word in word.one_away_words:
            if close_word == word2:
                break
            if not close_word.visited:
                s.append(close_word)
                close_word.prev = word
                close_word.visited = True
    word = word2
    output = deque()
    while word:
        output.appendleft(word)
        word = word.prev
    return output
