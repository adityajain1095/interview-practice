from pprint import pprint

#swap two numbs without temp
def moderate_one(a,b):
    # a = b-a
    # b = b-a
    # a = a+b
    a = a^b
    b = a^b
    a = a^b
    '''
    a = 7, b = 4
    a = 4-7 = -3
    b = 4-(-3) = 7
    a = 7 - 3 = 4
    '''
    return a,b
#tic tac two winner
def moderate_two(matrix):
    row = len(matrix)
    col = len(matrix[0])
    for i in range(row):
        for j in range(col):
            player = matrix[i][j]
            if moderate_two_helper(matrix,i,j,row,col):
                print(player, ' won')
                print(i,j)
                return player
    print('no one won')
    return None

def moderate_two_helper(matrix,i,j,row,col):
    if (i == 0 and j == 0) or (i == 0 and j == col-1) or (i == row-1 and j == 0) or (i == row-1 and j == col-1):
        return False
    elif i == 0 or i == row-1:
        if matrix[i][j] == matrix[i][j+1] and matrix[i][j] == matrix[i][j-1]:
            return True
        return False
    elif j == 0 or j == col-1:
        if matrix[i][j] == matrix[i+1][j] and matrix[i][j] == matrix[i+1][j]:
            return True
        return False
    else:
        if matrix[i][j] == matrix[i-1][j] and matrix[i][j] == matrix[i+1][j]:
            return True
        if matrix[i][j] == matrix[i][j-1] and matrix[i][j] == matrix[i][j-1]:
            return True
        if matrix[i][j] == matrix[i-1][j-1] and matrix[i][j] == matrix[i+1][j+1]:
            return True
        if matrix[i][j] == matrix[i-1][j+1] and matrix[i][j] == matrix[i+1][j-1]:
            return True
        return False

# matrix = [[0,0,1],[1,0,1],[0,1,1]]
# print(moderate_two(matrix))
'''
0 0 1
1 0 1
0 1 1
'''

#return max with no if else
def moderate_four(a,b):
    c = a-b
    k = c >> 31 and 1
    return a - (k*c)

#Given an integer between 0 and 999,999, print an English phrase that describes the
#integer (eg, “One Thousand, Two Hundred and Thirty Four”).

def moderate_six(num):
    n = len(str(num))
    string = ''
    if n < 3:
        if n == 3:
            string += 'hundred' + 'tens' + 'one'
        elif n == 2:
            if str(num)[n-1] < '2':
                string += 'special'
            else:
                string += 'tens' + 'one'
        else:
            string += 'one'
    else:
        for i in range(0,n,3):
            k = i+3
            if k > n:
                if k - n == 1:
                    k = 2
                else:
                    k = 1
            k = k%3
            if k == 0:
                string += 'hundred' + 'tens' + 'one'
            elif k == 2:
                if str(num)[n-1] < '2':
                    string += 'special'
                else:
                    string += 'tens' + 'one'
            else:
                string += 'one'
    return string

def moderate_seven(arr):
    maximum = current = 0
    if not arr:
        return 0
    maximum_arr = current_arr = []
    for i in range(len(arr)):
        if current + arr[i] > 0:
            current += arr[i]
            current_arr.append(arr[i])
            if current > maximum:
                maximum_arr = current_arr.copy()
                print(maximum_arr)
                maximum = current
        else:
            current_arr = []
            current = 0
    return maximum, maximum_arr

def moderate_seven_adjusted(arr):
    if not arr:
        return 0
    negatives = True
    biggest_negative = float('-inf')
    for x in arr:
        if x > 0:
            negatives = False
            break
        biggest_negative = max(biggest_negative,x)
    if negatives:
        return biggest_negative
    current = maximum = 0
    for x in arr:
        current = max(current+x,0)
        maximum = max(current,maximum)
    return maximum


arr = []
print(moderate_seven(arr))
print(moderate_seven_adjusted(arr))
