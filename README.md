# KMV

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python", width="120" height="20"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license",width="120" height="20"></a> &nbsp;
</p>


---

**KMV**  model is a method established by KMV Company of San Francisco in 1997 to estimate the default probability of borrowing enterprises. The model holds that , is determined by the market value of assets in the case of a given liability. However, assets are not actually traded in the market, and assets cannot be directly observed. To this end, the model considers the bank's loan problem from the perspective of the owner of the borrowing business and considers the problem of loan repayment. On the maturity date, if the market value of the company's assets is higher than the company's debt value (default point), the company's equity value is the difference between the company's asset market value and the debt value; if the company's asset value is lower than the company's debt value at this time, Then the company sells all assets to repay debts, and the equity value becomes zero.

Check out **http://finance.sxy.suda.edu.cn/_upload/article/files/d7/d0/10c54f454f259488ed38016fa9e9/ee01256b-94e8-43d8-a19b-1226e6f71d2e.pdf** to look at all the functions the tool supports or continue below for some brief examples.


## Setup

> Install from github

```shell
$ pip install git+https://github.com/leetung0718/KMV.git
```

## Example
> Import Module
```python
from kmv import KMV
```
> Initialize Model
```python
model = KMV()
```
> Sample Input
```python
r = 0.012 # risk free rate
T = 1 # duration, 1 year
L = 712788068 # liabilities
E = 13793196 # market value
V = 0.260344488 # equity volatility

model.fit(r, L, E, V, T)
```
> View Result
```python
model.result()
#
(52.060369711520856, 0.005001069830276989)
```
> View Summary
```python
model.summary()
#
Asset Coefficient        5.206037e+01
Asset Market Value       7.180789e+08
Asset Volatility         5.001070e-03
Default Distance KMV     1.473288e+00
Default Distance Merton  3.875728e+00
ND1                      1.000000e+00
ND2                      1.000000e+00
```