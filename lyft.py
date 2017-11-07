'''
all subsets of string
lca of binary tree
tinyurl
lru cache
lonely island count
longest branch in tree
second max in array
'''
from pprint import pprint
import random

def subsets_of_string(string,k):
    perm = ['']
    output = set()
    if k == 0:
        return output
    if k == 1:
        output = set(list(string))
        output.add('')
        return output
    string_list = list(string)
    for char in string_list:
        # perm.extend([x+char for x in perm])
        l = len(perm)
        for i in range(l):
            word = perm[i] + char
            if len(word) == k:
                output.add(word)
            perm.append(word)
    # for subset in perm:
    #     if len(subset) == k:
    #         output.add(subset)
    return output

# print(subsets_of_string('hello',3))

def permutationsHelper(string, array, step=0):
    if step == len(string):
        return array.append(''.join(string))
    for i in range(step,len(string)):
        copy = string[:]
        copy[step], copy[i] = string[i], string[step]
        permutationsHelper(copy,array,step+1)

def permutations(string):
    arr = []
    permutationsHelper(list(string),arr)
    return arr

def permutationsStack(string):
    arr = []
    s = []
    string = list(string)
    s.append((string,0))
    while len(s) != 0:
        x = s.pop()
        permutation = x[0]
        step = x[1]
        if step == len(string):
            arr.append(''.join(permutation))
        for i in range(step,len(string)):
            copy = permutation[:]
            copy[step], copy[i] = permutation[i], permutation[step]
            s.append((copy,step+1))
    return arr

# pprint(len(permutations('hellohello')))
# pprint(len(permutationsStack('hellohello')))

class Node(object):

    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

'''
node1 left node2 right vice versa
node1 par, node2 child vice versa

'''
def lowestCommonAncestor(self, root, node1, node2):
    if not root:
        return None
    if root == node1 or root == node2:
        return root
    left = self.lowestCommonAncestor(root.left,node1,node2)
    right = self.lowestCommonAncestor(root.right,node1,node2)
    if right and left:
        return root
    if right:
        return right
    return left

'''
autoincrement tinyurl
first input as sharding key
do it by country
do it when it fills up

sql b/c need autoincrement

'''
def maxDepth(root,size=0):
    if not root:
        return size
    left = maxDepth(root.left,size+1)
    right = maxDepth(root.right,size+1)
    return max(right,left)

def second_max(arr):
    if len(arr) <= 1:
        return None
    first_max, second_max = max(arr[0],arr[1]), min(arr[0],arr[1])
    for i in range(2,len(arr)):
        x = arr[i]
        if x >= first_max:
            second_max = first_max
            first_max = x
        else:
            if x > second_max:
                second_max = x
    return second_max

arr = [1,12,3,4,5,6,7,8,9,10,11,2]
# random.shuffle(arr)
# print(arr)
# print(second_max(arr))


'''
convert to set

[1,2,3,4]

[2,3,4]
[1,3,4]
[1,2,4]
[1,2,3]

'''
def subsets_of_string(s):
    output = ['']
    l = list(s)
    l.sort()
    s = ''.join(l)
    for x in s:
        new_addition = []
        for y in output:
            new_addition.append(x+y)
        output.extend(new_addition)
    return output

print(subsets_of_string('1112'))

def permutationsBetter(nums):
    permutations = [[]]
    nums.sort()
    for num in nums:
        new_permutations = []
        for perm in permutations:
            for i in range(len(perm)+1):
                new_permutations.append(perm[:i]+[num]+perm[i:])
                if i < len(perm) and num == perm[i]:
                    break
        permutations = new_permutations
    return permutations

#test
arr = [1,1,1,2]
print(permutationsBetter(arr))

def permutations(nums):
    output = []
    used = set()
    perm_helper(nums,output,[],used)
    return output

def perm_helper(nums, output, path, used):
    if len(path) == len(nums):
        output.append(path)
    else:
        for i in range(len(nums)):
            if i in used or (i > 0 and nums[i] == nums[i-1] and i-1 in used):
                continue
            used.add(i)
            perm_helper(nums,output,path+[nums[i]],used)
            used.remove(i)
            
print(permutations(arr))

def subsets_dup(nums):
    output = []
    path = []
    nums.sort()
    subsets_dup_help(nums,output,path,0)
    return output
    
def subsets_dup_help(nums,output,path,index):
    output.append(path)
    for i in range(index,len(nums)):
        if i > index and nums[i] == nums[i-1]:
            continue
        subsets_dup_help(nums,output,path+[nums[i]],i+1)
        
print(subsets_dup(arr))
    
def inorder(root):
    current = root
    s = []
    output = []
    while len(s) or current:
        while current:
            s.append(current)
            current = current.left
        node = s.pop()
        output.append(node)
        current = node.right
        
    return output
        

    
