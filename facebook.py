
def smallest_from_n(n):
    #is it a one? yes
    #is it negative? no
    #same amount of digits? or can use repeats
    #nlogn
    if n//10 == 0:
        return None
    arr = []
    while n:
        n, m = divmod(n,10)
        arr.append(m)
    arr = arr[::-1]
    sorted_arr = sorted(arr)
    breakable = False
    minimum = sorted_arr[0]
    for i in range(len(arr)-1,-1,-1):
        digit = arr[i]
        if digit == minimum:
            continue
        for j in range(len(sorted_arr)-1,-1,-1):
            if sorted_arr[j] < digit:
                arr[i] = sorted_arr[j]
                breakable = True
                break
        if breakable:
            break
    if not breakable:
        return None
    x = 0
    for i in range(len(arr)):
        x *= 10
        x += arr[i]
    return x

def anagram_checker(string1, string2):
    first_map = {}
    for char in string1:
        val = first_map.get(char,0)
        first_map[char] = val + 1
    for char in string2:
        val = first_map.get(char,0)
        first_map[char] = val - 1
        if first_map[char] == -1:
            return False
    return True

from functools import cmp_to_key


def comparison(string1,string2):
    if sorted(string1) < sorted(string2):
        return -1
    elif sorted(string1) == sorted(string2):
        if string1 < string2:
            return -1
        if string1 == string2:
            return 0
    return 1

def list_sort_anagram(arr):
    return sorted(arr, key=cmp_to_key(comparison))

# print(list_sort_anagram(['cba','abc','hel','tim','leh']))

x = [1,2,3]
y = [4,5,6]
zipped = zip(x,y)
print(zipped)
x1 ,y2 = zip(*zip(x,y))

# print(sum([i[0]*i[1] for i in zipped]))

x = ['a','b','a']

output = ['']

for y in x:
    l = len(output)
    for i in range(l):
        output.append(output[i]+y)

print(set(output))

output = []
def subset_recur(string_list, output):
    pass


def longest_sequence(nums,target):
    if not nums:
        return []
    result = []
    nums.sort()
    longest_sequence_helper(nums,0,target,[],result)
    return result

def longest_sequence_helper(nums,index,target,path,result):
    if target >= 0:
        if target == 0:
            result.append(path)
        else:
            for i in range(index, len(nums)):
                longest_sequence_helper(nums,i,target-nums[i],path+[nums[i]],result)

nums = [1,2,3,6]
target = 7
print(longest_sequence(nums,target))



def num_2_3(nums):
    result = []
    num_2_3_helper(nums,0,1,result,0)

def num_2_3_helper(nums,index,multiplication,result,number):
    if number == 3:
        result.append(multiplication)
    else:
        if number == 2:
            result.append(multiplication)
            for i in range(index,len(result)):
                num_2_3_helper(nums,i,multiplication*nums[i],result,number+1)
