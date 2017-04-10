from math import isnan
from per_iter_funcs import *

base = "/home/arapat/workspace/research/boosting/experiments/results/splice-site/auto/"

labels = ["Training auPRC", "Testing auPRC", "Testing (ref) auPRC",
          "Training average score =", "Training average score (positive)",
          "Training average score (negative)",
          "Testing average score =", "Testing average score (positive)",
          "Testing average score (negative)",
          "Testing (ref) average score =", "Testing (ref) average score (positive)",
          "Testing (ref) average score (negative)",
          "Effective count", "Positive effective count", "Negative effective count"]

lines, rollback1 = extractLines(base + "result-010.txt", labels)
nums = extractNums(lines)
joint1 = nums

lines, rollback2 = extractLines(base + "result-002.txt", labels)
nums = extractNums(lines)
joint2 = nums

for nums in [joint1, joint2]:
    for idx in [2, 9, 10, 11]:
        last = nums[idx][0]
        for j in range(len(nums[idx])):
            if isnan(nums[idx][j]):
                nums[idx][j] = last
            else:
                last = nums[idx][j]

