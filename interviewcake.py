from pprint import pprint
from itertools import *

def apple_stocks(stocks):
    '''
    Write an efficient function that takes stock_prices_yesterday and returns the best profit
    I could have made from 1 purchase and 1 sale of 1 Apple stock yesterday.
    '''
    '''
    implementation. step by step. keep track of min price before. then make max be current-min
    then change min based off current and min

    '''
    if not len(stocks):
        raise ValueError('none input')
    if len(stocks) < 2:
        raise ValueError('imcomplete stock data')
    _min = stocks[0]
    best_profit = stocks[1] - _min
    for stock in islice(stocks,1,None):
        best_profit = max(stock-_min,best_profit)
        _min = min(_min,stock)
    return best_profit

stocks = [10, 7, 5, 8, 11, 8 ,14,18]

def apple_stocks_2(stocks):
    '''
    buy/sell in one subsequence
    '''

    current = best_profit = 0
    for stock in stocks:
        current = max(current+stock,0)
        best_profit = max(current.best_profit)
    return best_profit

def max_apple_stocks(stocks):
    '''
    unlimited purchases/sells
    '''
    current = best_profit = 0
    if not len(stocks):
        raise ValueError('none input')
    if len(stocks) < 2:
        raise ValueError('imcomplete stock data')
    for i in range(1,len(stocks)):
        best_profit += max(0,stocks[i]-stocks[i-1])
    return best_profit

def product_of_all_other_numbers(arr):
    '''
    Start with a brute force solution, look for repeat work in that solution, and modify it to only do that work once.
    '''
    output = [1]*len(arr)
    prev_product = 1
    for i in range(len(arr)):
        output[i] *= prev_product
        prev_product *= arr[i]
    prev_reverse_product = 1
    for i in range(len(arr)-1,-1,-1):
        output[i] *= prev_reverse_product
        prev_reverse_product *= arr[i]
    return output

product_list = [1, 7, 3, 4]
# print(product_of_all_other_numbers(product_list))

def highest_of_three(arr):
    '''
    Given a list of integers, find the highest product you can get from three of the integers.

    notes:
    possibilities: all positive high numbers
                    two negative and positive
    keep track of two highest, two lowest, and max product
    '''
    if len(arr) < 3:
        raise ValueError('need at least 3 in list')
    max_product = arr[0]*arr[1]*arr[2]
    sorted_arr = sorted(arr[:3])
    highest_num = sorted_arr[2]
    lowest_num = sorted_arr[0]
    second_highest_num = second_lowest_num = sorted_arr[1]
    for current in islice(arr, 3, None):
        max_product = max(max_product,highest_num*second_highest_num*current,lowest_num*second_lowest_num*current)
        if current > highest_num:
            highest_num,second_highest_num = current,highest_num
        elif current > second_highest_num:
            second_highest_num = current

        if current < lowest_num:
            lowest_num,second_lowest_num = current,lowest_num
        elif current < second_lowest_num:
            second_lowest_num = current
        max_product = max(max_product,highest_num*lowest_num*second_lowest_num)
    return max_product

product_list = [1, 7, 3, 4,-5,2,4,4,3,-12,3]
# print(highest_of_three(product_list))

def merge_meeting_times(meetings):
    if not meetings:
        raise ValueError('meetings list is empty')
    meetings_sorted = sorted(meetings)
    output = []
    lower_bound = meetings[0][0]
    upper_bound = meetings[0][1]
    for i in range(1,len(meetings)):
        if meetings_sorted[i][0] > upper_bound:
            output.append((lower_bound,upper_bound))
            lower_bound = meetings_sorted[i][0]
            upper_bound = meetings_sorted[i][1]
        else:
            if upper_bound < meetings_sorted[i][1]:
                upper_bound = meetings_sorted[i][1]
    output.append((lower_bound,upper_bound))
    return output

meetings = [(0, 1), (3, 5), (4, 8), (10, 12), (9, 10)]
# print(merge_meeting_times(meetings))

class MaxStack:

    # initialize an empty list
    def __init__(self):
        self.items = []
        self.maximum = []
    # push a new item to the last index
    def push(self, item):
        if self.items:
            if item > self.maximum[-1]:
                self.maximum.append(item)
            else:
                self.maximum.append(self.maximum[-1])
        else:
            self.maximum = [item]
        self.items.append(item)

    # remove the last item
    def pop(self):
        # if the stack is empty, return None
        # (it would also be reasonable to throw an exception)
        if not self.items:
            return None
        self.maximum.pop()
        return self.items.pop()

    # see what the last item is
    def peek(self):
        if not self.items:
            return None
        return self.items[-1]

    def get_max(self):
        return self.maximum[-1]
