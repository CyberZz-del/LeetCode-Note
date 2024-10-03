[TOC]

# 169.多数元素

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

**示例 1：**

```
输入：nums = [3,2,3]
输出：3
```

**示例 2：**

```
输入：nums = [2,2,1,1,1,2,2]
输出：2
```

 

**提示：**

- `n == nums.length`
- `1 <= n <= 5 * 104`
- `-109 <= nums[i] <= 109`

 

**进阶：**尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。

**解法1**：哈希表

可以用哈希表快速统计每个元素出现的次数

```py
import collections

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get)
```

> [`Counter`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter) 是 [`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict) 的子类，用于计数 [hashable](https://docs.python.org/zh-cn/3/glossary.html#term-hashable) 对象。它是一个多项集，元素存储为字典的键而它们的计数存储为字典的值。计数可以是任何整数，包括零或负的计数值。[`Counter`](https://docs.python.org/zh-cn/3/library/collections.html#collections.Counter) 类与其他语言中的 bag 或 multiset 很相似。
>
> 它可以通过计数一个 *iterable* 中的元素来初始化，或用其它 *mapping* (包括 counter) 初始化：

时间复杂度O(n) 空间复杂度O(n)

---

# 121.买卖股票的最佳时机

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

 

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**示例 2：**

```
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```

**提示：**

- `1 <= prices.length <= 105`
- `0 <= prices[i] <= 104`

```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = 100000
        maxprofit = 0
        for i in range(len(prices)):
            profit = prices[i] - minprice
            if minprice > prices[i]:
                minprice = prices[i]
            if profit > maxprofit:
                maxprofit = profit
        return maxprofit    

```

**动态规划：**

假设现在第i天，前i天的最低价格与前i-1天的最低价格有关 `minprice[i] = min(minprice[i-1], price[i])` ，第i天出售股票的最高收益与前i-1天的最第价格有关 `maxprofit = max(maxprofit, price[i] - minprice[i-1])` 

时间复杂度O(n)，空间复杂度O(1)

---

# 55.跳跃游戏

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

**示例 2：**

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

 

**提示：**

- `1 <= nums.length <= 104`
- `0 <= nums[i] <= 105`

**贪心算法**，遍历数组`nums`，维护当前可到达的最远距离 `maxlen = max(maxlen, i+nums[i])`

```py
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        maxlen = 0
        for i in range(len(nums)):
            if i > maxlen:
                break
            if(maxlen < i+nums[i]):
                maxlen = i+nums[i]
        if maxlen >= len(nums)-1:
            return True
        else:
            return False
```

变体：给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:

- `0 <= j <= nums[i]` 
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

贪心算法：维护当前能够到达的最大下标，称为边界

```py
class Solution:
    def jump(self, nums: List[int]) -> int:
        jump, mlen, end=0, 0, 0
        for i in range(len(nums)-1):
            
            if mlen >= i:
                mlen = max(mlen, i+nums[i])
                if i == end:
                    jump+=1
                    end = mlen
        return jump
```



# 380.O(1)时间插入，删除，获取随机元素

实现`RandomizedSet` 类：

- `RandomizedSet()` 初始化 `RandomizedSet` 对象
- `bool insert(int val)` 当元素 `val` 不存在时，向集合中插入该项，并返回 `true` ；否则，返回 `false` 。
- `bool remove(int val)` 当元素 `val` 存在时，从集合中移除该项，并返回 `true` ；否则，返回 `false` 。
- `int getRandom()` 随机返回现有集合中的一项（测试用例保证调用此方法时集合中至少存在一个元素）。每个元素应该有 **相同的概率** 被返回。

你必须实现类的所有函数，并满足每个函数的 **平均** 时间复杂度为 `O(1)` 。

 	**示例：**

```
输入
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
输出
[null, true, false, true, 2, true, false, 2]
```



**解法：**

数组可以在O(1)的时间内获取随机元素，但是由于无法在O(1)时间内判断元素是否在数组中，所以不能在O(1)时间内插入和删除元素

哈希表可以在O(1)时间内插入和删除元素，但无法在O(1)时间内获取随机元素

将数组与哈希表结合使用，同时维护一个数组和一个哈希表，即可达到效果：

```py
from random import choice
class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.indices = {}

    def insert(self, val: int) -> bool:
        if val in self.indices: #O(1)时间内查找--哈希表
            return False
        self.indices[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.indices: #O(1)时间内查找--哈希表
            return False
        id = self.indices[val]
        self.nums[id] = self.nums[-1]
        self.indices[self.nums[id]] = id
        self.nums.pop()
        del self.indices[val]
        return True

    def getRandom(self) -> int:
        return choice(self.nums) #O(1)时间内获取--数组
```

# 238.除自身以外数组的乘积

给你一个整数数组 `nums`，返回 数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内。

请 **不要使用除法，**且在 `O(n)` 时间复杂度内完成此题。

 

**示例 1:**

```
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
```

**示例 2:**

```
输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]
```

解法：左右乘积列表

正序和倒序遍历`nums`数组 ，维护两个列表 `Fronlist` 和 `Baclist` 分别保存前i个元素的积和后i个元素的积，然后`answer`的元素直接调用对应的 `Fronlist` 和 `Baclist` 相乘即可

```py
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        Fronlist = []
        mul = 1 
        for i in range(len(nums)):
            mul = mul * nums[i]
            Fronlist.append(mul)

        Baclist = []
        mul = 1
        for i in range(len(nums)):
            mul = mul * nums[len(nums)-i-1]
            Baclist.append(mul)

        answer = []
        for i in range(len(nums)):
            if i==0:
                answer.append(Baclist[len(nums)-2])
                continue
            if i==len(nums)-1:
                answer.append(Fronlist[len(nums)-2])
                continue
            answer.append(Fronlist[i-1]*Baclist[len(nums)-i-2])
        return answer

```

# 134.加油站

在一条环路上有 `n` 个加油站，其中第 `i` 个加油站有汽油 `gas[i]` 升。

你有一辆油箱容量无限的的汽车，从第 `i` 个加油站开往第 `i+1` 个加油站需要消耗汽油 `cost[i]` 升。你从其中的一个加油站出发，开始时油箱为空。

给定两个整数数组 `gas` 和 `cost` ，如果你可以按顺序绕环路行驶一周，则返回出发时加油站的编号，否则返回 `-1` 。如果存在解，则 **保证** 它是 **唯一** 的。

 

**示例 1:**

```
输入: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
输出: 3
解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。
```

解法：贪心算法

首先检查第 0 个加油站，并试图判断能否环绕一周；如果不能，就从第一个无法到达的加油站开始继续检查。

原理：如果x不能到达y，那么x到y之间的所有加油站都不能到达y

```py
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        lenth = len(gas)
        lis = [0]*lenth
        for i in range(lenth):
            lis[i] = gas[i] - cost[i]

        i = 0
        while(i<lenth):
            if lis[i] >= 0:
                p = 0
                j = 0
                while(j<lenth and p >= 0): #从第i个加油站开始跑
                    p+=lis[(i+j)%lenth]
                    if p<0: #跑不动了
                        break
                    j+=1
                if j == lenth: #跑到头了
                    return i
                else: #没跑到头
                    i=i+j #接下来要从第一个没跑到的加油站开始跑
                    continue
            else:
                i+=1
            
        return -1
```

# 135.分发糖果

`n` 个孩子站成一排。给你一个整数数组 `ratings` 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

- 每个孩子至少分配到 `1` 个糖果。
- 相邻两个孩子评分更高的孩子会获得更多的糖果。

请你给每个孩子分发糖果，计算并返回需要准备的 **最少糖果数目** 。

 

**示例 1：**

```
输入：ratings = [1,0,2]
输出：5
解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
```

解法：左右遍历

- 左规则：如果 `rating[i-1]<rating[i]` 那么 `left[i]=left[i-1]+1` 否则 `left[i]=1`
- 右规则：如果 `rating[i+1]<rating[i]` 那么 `right[i]=right[i+1]+1` 否则 `right[i]=1`

最终分发糖果的结果序列：`lis[i] = max(left[i],right[i])`

```py
class Solution:
    def candy(self, ratings: List[int]) -> int:
        
        n = len(ratings)
        if n == 1: #特殊情况
            return 1
        left = [0]*n
        right = [0]*n
        for i in range(n): #左规则
            if i>0 and ratings[i-1]<ratings[i]:
                left[i] = left[i-1] + 1
            else:
                left[i] = 1
        for i in reversed(range(n)): #右规则
            if i < n-1 and ratings[i+1]<ratings[i]:
                right[i] = right[i+1] + 1
            else:
                right[i] = 1
        lis = [0]*n
        for i in range(n):
            lis[i] = max(left[i],right[i])
        return sum(lis)
```

# 42.接雨水

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 

**示例 1：**

![img](./assets/rainwatertrap.png)

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

解法1：暴力

从下到上检查二维表每一个点，只要两边有墙，下面有墙或者水，则满足条件

结果：内存溢出

```py
class Solution:
    
    def trap(self, height: List[int]) -> int:
        water = 0
        block = []
        maxhight = max(height)
        n = len(height)
        for i in range(n):
            lis = [1]*height[i]
            while(len(lis)<maxhight):
                lis.append(0)
            block.append(lis)
        for j in range(maxhight):
            for i in range(n):
                if block[i][j] == 0:
                    if j == 0:
                        flag1, flag2 = False, False
                        for i1 in range(i):
                            if block[i1][j] == 1:
                                flag1 = True
                        for i2 in range(i+1, n):
                            if block[i2][j] == 1:
                                flag2 = True
                        if flag1 and flag2:
                            print(f"({i}, {j})")
                            block[i][j] = 2
                            water += 1
                        else:
                            block[i][j] = -1
                    else:
                        if block[i][j-1] == -1:
                            block[i][j] = -1
                            continue
                        elif block[i][j-1] == 1 or block[i][j-1] == 2:
                            flag1, flag2 = False, False
                            for i1 in range(i):
                                if block[i1][j] == 1:
                                    flag1 = True
                            for i2 in range(i+1, n):
                                if block[i2][j] == 1:
                                    flag2 = True
                            if flag1 and flag2:
                                print(f"({i}, {j})")
                                block[i][j] = 2
                                water += 1
                            else:
                                block[i][j] = -1
                                continue
        return water  
```

解法2：

还是暴力（笑

先找到一个“最高峰”，再寻找一个“次高峰”，“次高峰”和“最高峰”之间不高于”次高峰“的空气格子都能装雨水，计算完后再换一个“次高峰”，重复以上步骤，直到没有“次高峰”。

```py
class Solution:
    
    def trap(self, height: List[int]) -> int:
        if len(set(height)) == 1:
            return 0
        if all(height[i] <= height[i + 1] for i in range(len(height) - 1)) or \
        all(height[i] >= height[i + 1] for i in range(len(height) - 1)):
            return 0    #单调的数组直接返回0
        block = []
        n = len(height)
        lised = [-1]*n
        for i in range(n):
            block.append({"index":i, "height":height[i]})
        block.sort( key = lambda x : x["height"])
        highest = block.pop() #弹出最高峰
        index_highest = highest["index"]
        
        while len(block) > 0:
            current_height = block.pop()   #弹出次高峰直到空栈
            index_cur = current_height["index"]
            if lised[index_cur] >= 0:
                continue
            height_cur = current_height["height"]
            if index_cur > index_highest:
                for i in range(index_highest+1, index_cur, 1):
                    if lised[i] < 0:
                        lised[i] = height_cur - height[i]
                        if lised[i] < 0:
                            lised[i] = 0
                        lised[index_cur] = 0
            else:
                for i in range(index_highest-1, index_cur, -1):
                    if lised[i] < 0:
                        lised[i] = height_cur - height[i]
                        if lised[i] < 0:
                            lised[i] = 0
                        lised[index_cur] = 0
        lised[index_highest] = 0
        lised[0]=0
        lised[n-1] = 0
        assert(-1 not in lised)
        return sum(lised)   
```

解法3：动态规划（官解

创建两个长度为 n 的数组 $leftMax 和 rightMax$。对于 $0≤i<n，leftMax[i]$ 表示下标 i 及其左边的位置中，$height$ 的最大高度，r$ightMax[i]$ 表示下标 i 及其右边的位置中，height 的最大高度。

显然，$leftMax[0]=height[0]$，$rightMax[n−1]=height[n−1]$。两个数组的其余元素的计算如下：

当 $1≤i≤n−1$ 时，$leftMax[i]=max(leftMax[i−1],height[i])；$

当 $0≤i≤n−2$ 时，$rightMax[i]=max(rightMax[i+1],height[i])。$

因此可以正向遍历数组 height 得到数组 leftMax 的每个元素值，反向遍历数组 height 得到数组 rightMax 的每个元素值。

在得到数组 leftMax 和 rightMax 的每个元素值之后，对于 0≤i<n，下标 i 处能接的雨水量等于 $min(leftMax[i],rightMax[i])−height[i]$。遍历每个下标位置即可得到能接的雨水总量。

动态规划做法可以由下图体现。

![fig1](./assets/1.png)

```py
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        
        n = len(height)
        leftMax = [height[0]] + [0] * (n - 1)
        for i in range(1, n):
            leftMax[i] = max(leftMax[i - 1], height[i])

        rightMax = [0] * (n - 1) + [height[n - 1]]
        for i in range(n - 2, -1, -1):
            rightMax[i] = max(rightMax[i + 1], height[i])

        ans = sum(min(leftMax[i], rightMax[i]) - height[i] for i in range(n))
        return ans
```

解法4：单调栈

除了计算并存储每个位置两边的最大高度以外，也可以用单调栈计算能接的雨水总量。

维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 height 中的元素递减。

从左到右遍历数组，遍历到下标 i 时，如果栈内至少有两个元素，记栈顶元素为 top，top 的下面一个元素是 left，则一定有 $height[left]≥height[top]$。如果 $height[i]>height[top]$，则得到一个可以接雨水的区域，该区域的宽度是 i−left−1，高度是 $min(height[left],height[i])−height[top]，$根据宽度和高度即可计算得到该区域能接的雨水量。

为了得到 left，需要将 top 出栈。在对 top 计算能接的雨水量之后，left 变成新的 top，重复上述操作，直到栈变为空，或者栈顶下标对应的 height 中的元素大于或等于 $height[i]。$

在对下标 i 处计算能接的雨水量之后，将 i 入栈，继续遍历后面的下标，计算能接的雨水量。遍历结束之后即可得到能接的雨水总量。

```py
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        stack = list()
        n = len(height)
        
        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                left = stack[-1]
                currWidth = i - left - 1
                currHeight = min(height[left], height[i]) - height[top]
                ans += currWidth * currHeight
            stack.append(i)
        
        return ans
```

> `enumerate(iterable, start=0)` 
>
> 返回一个枚举对象。*iterable* 必须是一个序列，或 [iterator](https://docs.python.org/zh-cn/3/glossary.html#term-iterator)，或其他支持迭代的对象。 [`enumerate()`](https://docs.python.org/zh-cn/3/library/functions.html#enumerate) 返回的迭代器的 [`__next__()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#iterator.__next__) 方法返回一个元组，里面包含一个计数值（从 *start* 开始，默认为 0）和通过迭代 *iterable* 获得的值。
>
> \>>>
>
> ```py
> >>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
> >>> list(enumerate(seasons))
> [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
> >>> list(enumerate(seasons, start=1))
> [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
> ```
>
> 等价于:
>
> ```py
> def enumerate(iterable, start=0):
>     n = start
>     for elem in iterable:
>         yield n, elem
>         n += 1
> ```

# 13.罗马数字转整数

七个不同的符号代表罗马数字，其值如下：

| 符号 | 值   |
| ---- | ---- |
| I    | 1    |
| V    | 5    |
| X    | 10   |
| L    | 50   |
| C    | 100  |
| D    | 500  |
| M    | 1000 |

罗马数字是通过添加从最高到最低的小数位值的转换而形成的。将小数位值转换为罗马数字有以下规则：

- 如果该值不是以 4 或 9 开头，请选择可以从输入中减去的最大值的符号，将该符号附加到结果，减去其值，然后将其余部分转换为罗马数字。
- 如果该值以 4 或 9 开头，使用 **减法形式**，表示从以下符号中减去一个符号，例如 4 是 5 (`V`) 减 1 (`I`): `IV` ，9 是 10 (`X`) 减 1 (`I`)：`IX`。仅使用以下减法形式：4 (`IV`)，9 (`IX`)，40 (`XL`)，90 (`XC`)，400 (`CD`) 和 900 (`CM`)。
- 只有 10 的次方（`I`, `X`, `C`, `M`）最多可以连续附加 3 次以代表 10 的倍数。你不能多次附加 5 (`V`)，50 (`L`) 或 500 (`D`)。如果需要将符号附加4次，请使用 **减法形式**。

给定一个整数，将其转换为罗马数字。

解法：双指针

```py
class Solution:
    def f(self, s:str) -> int:
        assert(len(s)==1 or len(s)==2)
        if s=='I':
            return 1
        elif s=='V':
            return 5
        elif s=='X':
            return 10
        elif s=='L':
            return 50
        elif s=='C':
            return 100
        elif s=='D':
            return 500
        elif s=='M':
            return 1000
        elif s=='IV':
            return 4
        elif s=='IX':
            return 9
        elif s=='XL':
            return 40
        elif s=="XC":
            return 90
        elif s=='CD':
            return 400
        elif s=='CM':
            return 900
        return 0
    def romanToInt(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return self.f(s)
        i = 0
        j = 1
        num = 0
        while(i<n and j < n):
            if self.f(s[i:j+1]) == 0:
                num += self.f(s[i:j])
                i+=1
                j+=1
            else:
                num += self.f(s[i:j+1])
                i+=2
                j+=2
        if i < n:
            num += self.f(s[i:j])
        return num
```

# 12.整数转罗马数字

同上

解法：递归秒了

```py
class Solution:
    def intToRoman(self, num: int) -> str:
        if num == 0:
            return ""
        elif num == 1:
            return "I"
        elif num > 1 and num < 4:
            return "I" + self.intToRoman(num - 1)
        elif num == 4:
            return "IV"
        elif num == 5:
            return "V"
        elif num > 5 and num < 9:
            return "V" + self.intToRoman(num - 5)
        elif num == 9:
            return "IX"
        elif num == 10:
            return "X"
        elif num > 10 and num < 40:
            return "X"+self.intToRoman(num - 10)
        elif num == 40:
            return "XL"
        elif num > 40 and num < 50:
            return "XL"+self.intToRoman(num - 40)
        elif num == 50:
            return "L"
        elif num > 50 and num < 90:
            return "L" + self.intToRoman(num - 50)
        elif num == 90:
            return "XC"
        elif num > 90 and num < 100:
            return "XC" + self.intToRoman(num - 90)
        elif num == 100:
            return "C"
        elif num > 100 and num < 400:
            return "C"+ self.intToRoman(num - 100)
        elif num == 400:
            return "CD"
        elif num > 400 and num < 500:
            return "CD" + self.intToRoman(num-400)
        elif num == 500:
            return "D"
        elif num > 500 and num < 900:
            return "D" + self.intToRoman(num - 500)
        elif num == 900:
            return "CM"
        elif num > 900 and num < 1000:
            return "CM" + self.intToRoman(num - 900)
        elif num == 1000:
            return "M"
        elif num > 1000:
            return "M" + self.intToRoman(num - 1000)
```

~~python写代码真是太快了~~
