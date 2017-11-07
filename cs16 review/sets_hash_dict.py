#chapter5

'''
set()--> add, remove, contains/ is in, size, len,
list(enumerate(list,set,dictionary)) --> gives it by index, key}

set.update(t) --> add t to set
set.union(t) --> combine set and t
set.intersection(t) --> combine when both are in set and t
set.remove(x)
set.pop() --> random

union: O(m+n)
intersection: O(min(m,n))
difference: O(m)
issubset: O(m)

set is hashtable with dummy values in cpython
list is unhashable, need to be immutable state to be put in set

frozenset is set that is immutable

dict[key] = dict.get(key, value)
dict.items() --> tuples of (key, value)

hash table is open addressing

arr.sort() —> sorts actual arr

sorted(arr) —> returns deep copy of sorted arr, arr still not sorted

for key, value in dic_words.items(): --> to go through key,value of dictionary

'''

print(sorted('hello'))

def jumble(words):
	#return all words which have no permutation
	dic_words = {}
	for word in words:
		dic_words[''.join(sorted(word))] = dic_words.get(''.join(sorted(word)), 0)
		dic_words[''.join(sorted(word))] += 1
	output = []
	for key, value in dic_words.items():
		if value == 1:
			output.append(key)
	return output

# print(jumble(['ab', 'ba', 'an', 'na', 'bb', 'nn']))
