standard_prompt = '''Solve the word problem and give the final numeric answer.
Problem:
A box has 12 apples. You eat 5. How many remain?
Answer: 7
Problem:
Each notebook costs $3. You buy 4 notebooks and 2 pencils that cost $1 each. How much do you pay in total?
Answer: 14
Problem:
A class has 28 students. 12 are boys. How many are girls?
Answer: 16
Problem:
A train travels 60 miles per hour for 2.5 hours. How far does it travel?
Answer: 150
Problem:
A store sells packs of 6 markers. You need 5 packs and also buy a single marker for $1. Packs cost $4 each. How much do you spend?
Answer: 21
Problem:
{input}
Answer:'''

# 5-shot CoT (show steps; final turn ends with "Steps:" so the model continues)
cot_prompt = '''Solve step by step. Show your work, then conclude with "Final answer: <number>".
Problem:
Each sticker sheet has 8 stickers. You buy 3 sheets and then give 5 stickers to a friend. How many stickers do you have left?
Steps:
- 3 × 8 = 24
- 24 - 5 = 19
Final answer: 19
Problem:
A book has 150 pages. You read 20 pages each day for 4 days. How many pages remain?
Steps:
- 20 × 4 = 80
- 150 - 80 = 70
Final answer: 70
Problem:
A tank holds 45 liters of water. You drain 12 liters, then add 7 liters. How many liters are in the tank now?
Steps:
- 45 - 12 = 33
- 33 + 7 = 40
Final answer: 40
Problem:
A baker makes 36 cupcakes and packs them into boxes of 9. She eats 3 cupcakes. How many full boxes can she make?
Steps:
- 36 - 3 = 33
- 33 ÷ 9 = 3 remainder 6
- Full boxes = 3
Final answer: 3
Problem:
A car uses 4 gallons to travel 120 miles. How many miles per gallon does it get?
Steps:
- 120 ÷ 4 = 30
Final answer: 30
Problem:
{input}
Steps:
'''

# 1-shot propose (the ToT engine passes "Problem.../Steps so far..." in {input})
propose_prompt = '''Given the problem and the steps so far, output ONLY ONE line in ONE of these forms:
- ASSIGN: <symbol>=<number>      (e.g., ASSIGN: may=24)
- EQ: <simple equation>           (e.g., EQ: 28/2=14 or EQ: 12+24=36)

Rules:
- Exactly one line.
- No extra words, labels, bullets, or restating the problem.
- Do NOT give the final answer unless asked.

{input}
Next step:'''

# Value prompt: rate next-step promise (impossible / likely / sure)
value_prompt = '''Rate how promising the next step is for solving the problem.
Choose EXACTLY one: impossible, likely, sure.

Example 1
Problem:
A jar has 20 candies. You eat 5. How many are left?
Steps so far:
- 20 - 5 = 15
Next step quality: sure

Example 2
Problem:
There are 12 muffins. You buy 3 more packs of 4.
Steps so far:
- 12 + 3
Next step quality: likely

Example 3
Problem:
A train goes 60 mph for 2 hours.
Steps so far:
- 60 - 2 = 58
Next step quality: impossible

{input}
Your rating:'''

# Value prompt for a proposed FINAL answer
value_last_step_prompt = '''Judge whether the proposed FINAL answer is correct.
Choose EXACTLY one: impossible, likely, sure.

Example 1
Problem:
Each notebook costs $3. You buy 4.
Answer: 12
Your rating: sure

Example 2
Problem:
A class has 28 students. 12 are boys.
Answer: 15
Your rating: impossible

Example 3
Problem:
There are 45 liters of water. Drain 12, add 7.
Answer: 40
Your rating: sure

Problem:
{input}
Answer: {answer}
Your rating:'''