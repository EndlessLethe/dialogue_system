# introcution
A gallary of multi turn dialogue model

# Input
For each sample:  
Input - list_history and query represented by ids  
Output - 2 class softmax result

One sample one line, and k utterances is split by "\t"

# Expected final output:
shape = (n_sample, n_label) where n_label = 0 (not related) or 1(related)