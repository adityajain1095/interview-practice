from pprint import pprint
import random

matrix = [[1],[2]]
pprint(matrix)

def two_d_matrix_spiral(matrix):
    row_length = len(matrix)
    col_length = len(matrix[0])
    row_start = 0
    col_start = 0
    output = []
    while row_start < row_length and col_start < col_length:
        for i in range(col_start,col_length):
            output.append(matrix[row_start][i])
        row_start += 1

        for i in range(row_start,row_length):
            output.append(matrix[i][col_length-1])
        col_length -= 1

        for i in range(col_length-1,col_start-1,-1):
            if row_length != row_start:
                output.append(matrix[row_length-1][i])
        row_length -= 1

        for i in range(row_length-1,row_start-1,-1):
            if col_length != col_start:
                output.append(matrix[i][col_start])
        col_start += 1

    return output
print(two_d_matrix_spiral(matrix))

'''
Given two arrays A and B, Find the smallest subarray in A where we can find all of B.
sliding window, count and set of values of B

dfs with stack, bds with queue

subsequence of string: just go through arr
iteratively and check if you can complete word
in order

reverse string: reverse words, or create arr
, and append from back?

3 sum -> sort string, remove duplicates to make easier, then start at 0, index 1, and last, and do it

rank of the word
sort string,
then go through letter by letter and look
up all the possible words you can make that
are better, stop when at 0
then return count of words as rank
'''
