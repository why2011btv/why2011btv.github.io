[Excel表 - 刷题总结](https://docs.google.com/spreadsheets/d/191-GtkD55RkpTyiSsxEq7UD-28Vh9_kdeG3pZA2gO3Y/edit?usp=sharing)

[3, 528,]
# 数组
## 注意⚠️：循环不变量 左闭右开
- 704 (Binary Search)
  + <img width="736" alt="Screen Shot 2025-03-31 at 7 12 06 PM" src="https://github.com/user-attachments/assets/cf492557-2eea-45b4-be8d-708f5b78267c" />

  + <img width="766" alt="Screen Shot 2025-03-31 at 7 33 44 PM" src="https://github.com/user-attachments/assets/12dd9701-0aa9-4fcd-bf43-97c8a5789f92" />
  
- 27 (Remove Element)
  + <img width="697" alt="Screen Shot 2025-04-01 at 3 59 48 PM" src="https://github.com/user-attachments/assets/31d1c6b6-9dbb-4569-aa63-1ab82fa31d83" />

- 977 (sortedSquares)

- 209 (minSubArrayLen)
  + <img width="746" alt="Screen Shot 2025-04-01 at 11 35 25 PM" src="https://github.com/user-attachments/assets/1879c255-4d1f-4e7f-ad8e-9bdafdc7e371" />

- 59 (generateMatrix)
  + <img width="776" alt="Screen Shot 2025-04-02 at 8 59 47 PM" src="https://github.com/user-attachments/assets/668523fe-7912-4df1-a645-40f3151fd300" />
- 数组 7 区间和 （✅）
- 数组 8 开发商购买土地（嫌麻烦 未完成❌）

# 链表
- 203 (removeElements)
  + <img width="755" alt="Screen Shot 2025-04-02 at 9 52 41 PM" src="https://github.com/user-attachments/assets/35d42051-2b56-405b-b649-fdbc60e828e1" />
- 707 (MyLinkedList) 嫌麻烦未完成❌
- 206 (reverseList)
- 24 (swapPairs) 用时：16min
- 19 (removeNthFromEnd) 用时：9min
  + 但我的做法没有使用**双指针** 不够efficient 
- 160 (getIntersectionNode) 5min没思路 看答案加实现：14min
  + 注意到intersection node一定在末尾 所以先把尾端对齐
- **142** (detectCycle) 6min 没思路 看答案加实现：28min
  + <img width="937" alt="Screen Shot 2025-04-06 at 9 40 38 AM" src="https://github.com/user-attachments/assets/5a6fbd58-4384-4508-a9b6-ff88f20a1a02" />
  + <img width="769" alt="Screen Shot 2025-04-04 at 11 21 56 AM" src="https://github.com/user-attachments/assets/90459323-e160-40c1-8220-b51e75867be1" />

# 哈希表
## 数组实现
```Python
count = [0] * 101  # 键范围为0~100
nums = [1, 2, 2, 3]
for num in nums:
    count[num] += 1
print(count[2])  # 输出2
```
## set实现
```Python
seen = set()
nums = [1, 2, 3, 1]
for num in nums:
    if num in seen:
        print("重复元素：", num)
    seen.add(num)
```
## map实现
```Python
count = {}
nums = [1, 2, 2, 3]
for num in nums:
    count[num] = count.get(num, 0) + 1
print(count[2])  # 输出2
```

- 242
- 

