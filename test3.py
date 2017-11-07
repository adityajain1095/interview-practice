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

def combinationSum2(self, candidates, target):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, set(), res)
    return res

def dfs(self, nums, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        res.append(path)
        return
    for i in range(index, len(nums)):
        if i > index and nums[i] == nums[i-1]:
            continue
        if nums[i] in path:
            continue
        path.add(nums[i])
        self.dfs(nums, target-nums[i], i, path, res)

import collections

def window(s, t):
    need, missing = collections.Counter(t), len(t)
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



print(window('helloabbcdeffabcdefghijkl', 'acf'))
