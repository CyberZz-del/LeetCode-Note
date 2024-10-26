# 50. Pow(x, n)

实现 `pow(x, n)` ，即计算 `x` 的整数 `n` 次幂函数（即，$x^n$ ）。

**示例 1：**

```apach
输入：x = 2.00000, n = 10
输出：1024.00000
```

**示例 2：**

```apach
输入：x = 2.10000, n = 3
输出：9.26100
```

**示例 3：**

```apach
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
```

**提示：**

- `-100.0 < x < 100.0`
- `-2^31 <= n <= 2^31-1`
- `n` 是一个整数
- 要么 `x` 不为零，要么 `n > 0` 。
- `-10^4 <= x^n <= 10^4`

**解法一：** 暴力

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n > 0:
            ans = 1
            while n > 0:
                ans *= x
                n -= 1
            return ans
        elif n < 0:
            ans = 1
            while n < 0:
                ans /= x
                n += 1
            return ans
        else:
            return 1
```

虽然知道肯定不对，但是按流程还是先走一遍暴力。时间复杂`O(n)`，但是`n`的取值可以大的离谱，所以必然超时。

**解法二：** 快速幂

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def quickMul(N):
            if N == 0:
                return 1.0
            y = quickMul(N // 2)
            return y * y if N % 2 == 0 else y * y * x
        
        return quickMul(n) if n >= 0 else 1.0 / quickMul(-n)
```

一个简单的递归，$x^N=x^{N//2}*x^{N//2}$。这样复杂度就只有`O(logN)`了。
