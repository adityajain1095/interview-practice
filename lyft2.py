

import pprint



def permutations(nums):
    result = []
    nums.sort()
    used = set()
    permutationsHelper(nums, result, [], len(nums), used)
    return result
    
def permutationsHelper(nums, result, path, target_length, used):
    if len(path) == target_length:
        result.append(path)
    else:
        for i in range(len(nums)):
            if (i > 0 and nums[i] == nums[i-1] and i-1 not in used) or i in used:
                continue
            used.add(i)
            permutationsHelper(nums, result, path+[nums[i]], target_length, used)
            used.remove(i)
            
            
def permutationsBetter(nums):
    permutations = [[]]
    nums.sort()
    for num in nums:
        print(permutations)
        new_permutations = []
        for perm in permutations:
            for i in range(len(perm)+1):
                new_permutations.append(perm[:i]+[num]+perm[i:])
                if i < len(perm) and num == perm[i]:
                    print(i, perm[i])
                    break
        permutations = new_permutations
    return permutations

#test
arr = [2,2,1,1,1]
# print(permutations(arr))
# print(permutationsBetter(arr))
            
        
def lonely_island(matrix):
    
    row = len(matrix)
    col = len(matrix[0])
    memo = set()
    islands = 0
    for i in range(row):
        for j in range(col):
            islands += lonely_island_helper(matrix, i, j, memo)
    return islands

def lonely_island_helper(matrix, i, j, memo):
    island, directions = 0, [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
    if i >= 0 and j >= 0 and i < len(matrix) and j < len(matrix[0]):
        if (i, j) not in memo and matrix[i][j] == 1:
            island = 1
            memo.add((i,j))
            for dir in directions:
                lonely_island_helper(matrix,dir[0],dir[1],memo)
    return island

matrix = [[1, 1, 0, 0, 0],[0, 1, 0, 0, 1],[0, 0, 0, 1, 1],[0, 0, 0, 0, 0],[1, 0, 1, 0, 0]] 

print(lonely_island(matrix))
            