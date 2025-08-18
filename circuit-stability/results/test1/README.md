#TEST 1:

MAIN JSON
```python
{
      "Input" : "Input: 2 8 8 14\\nPossible next steps:\\n2 + 8 = 10 (left: 8 10 14)\\n8 / 2 = 4 (left: 4 8 14)\\n14 + 2 = 16 (left: 8 8 16)\\n2 * 8 = 16 (left: 8 14 16)\\n8 - 2 = 6 (left: 6 8 14)\\n14 - 8 = 6 (left: 2 6 8)\\n14 /  2 = 7 (left: 7 8 8)\\n14 - 2 = 12 (left: 8 8 12)\\nInput: 1 1 11 11\\nPossible next steps:\\n",
      "labels":["1 + 8 = 4 (right: 4 4 8)", "11 + 11 = 22 (left: 22 1 1)"]
}
```

Keeping the corrupted the same and running different cleans


run 1:

Clean prompt: Input+label[0]
corrupted_prompt = "Input: 2 8 8 14\\nPossible next steps:\\n2 + 8 = 10 (left: 8 10 14)\\n8 / 2 = 4 (left: 4 8 14)\\n14 + 2 = 16 (left: 8 8 16)\\n2 * 8 = 16 (left: 8 14 16)\\n8 - 2 = 6 (left: 6 8 14)\\n14 - 8 = 6 (left: 2 6 8)\\n14 /  2 = 7 (left: 7 8 8)\\n14 - 2 = 12 (left: 8 8 12)\\nInput: 2 6 11 13\\nPossible next steps:\\n5 % 13 = ?! (left: 6 9 56)" (A randomly chosen one)


run 2:

Clean prompt: Input+label[1]
corrupted_prompt = "Input: 2 8 8 14\\nPossible next steps:\\n2 + 8 = 10 (left: 8 10 14)\\n8 / 2 = 4 (left: 4 8 14)\\n14 + 2 = 16 (left: 8 8 16)\\n2 * 8 = 16 (left: 8 14 16)\\n8 - 2 = 6 (left: 6 8 14)\\n14 - 8 = 6 (left: 2 6 8)\\n14 /  2 = 7 (left: 7 8 8)\\n14 - 2 = 12 (left: 8 8 12)\\nInput: 2 6 11 13\\nPossible next steps:\\n5 % 13 = ?! (left: 6 9 56)" (A randomly chosen one)


from analysis.py
the skewness of each one is:
```bash
0_th_thought: skewness = -7.90251561326066
1_th_thought: skewness = 13.683598504784877

```

Note: 0th thought and 1st thought are in the respective order from json.