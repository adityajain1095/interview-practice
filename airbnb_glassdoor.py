
'''
Palindrome Pairs question - given a set of words, find all pairs of words that form a palindrome when concatenated.

menu problem:
public void orderCombination(int[] prices, int t) {
        orderCombinationHelper(prices, t, 0, new ArrayList());
    }

    private void orderCombinationHelper(int[] prices, int t, int i, List ordered) {
        if(t < 0 || i == prices.length) return;
        if(t == 0) System.out.println(ordered);
        ordered.add(i);
        orderCombinationHelper(prices, t - prices[i], i, ordered);
        orderCombinationHelper(prices, t - prices[i], i + 1, ordered);
        ordered.remove(ordered.size() - 1);
        orderCombinationHelper(prices, t, i + 1, ordered);
    }

iterator for array of arrays

1. Write a program to check a string is a palindrome.
2. Given a list of words, find the word pairs that when concatenated form a palindrome.
    ctci


'''
