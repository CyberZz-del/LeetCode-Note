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

