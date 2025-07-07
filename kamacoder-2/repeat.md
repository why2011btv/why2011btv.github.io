- UnionFind: how to implement **init, find, join**; can only be used to solve undirected graph
- Connectivity of directed graph: 
- BFS: when to add a node to visited (answer: add in the for loop)
- DFS: when to add a node to visited (answer: 1: at the beginning, use visited / coloring ([lc-130](https://leetcode.com/problems/surrounded-regions/description/), [lc-207](), [lc-210]() ; 2: backtracking)
- Topological Sort / Detect Cycle
- All subsequence of a string (from itertools import combinations; for l in range(1, min(max_len, len(chunk)) + 1); for comb in combinations(range(len(chunk)), l))
- Usage of lambda: indexed.sort(key=lambda x: (-x[1], x[0]))
  - sorted() 是函数，会返回新的列表，不修改原来的。
  - list.sort() 是方法，会在原地排序，返回 None。
```Python
words = ['apple', 'banana', 'pear']
sorted(words, key=len) # ['pear', 'apple', 'banana']
sorted(words, key=lambda x: x[-1])  # 按最后一个字母排序  ['banana', 'apple', 'pear']

nums = [3, 1, 4]
nums.sort() # [1, 3, 4]
```

- 无向图的连通分量
-
```Python
from sortedcontainers import SortedSet

days = SortedSet([1, 3, 4, 6, 9])

# 我想找 >= 5 的最小元素
d = days.bisect_left(5)
print(d)   # 输出 3，因为 days[3] = 6
print(days[d])  # 6
days.pop(d)
```
