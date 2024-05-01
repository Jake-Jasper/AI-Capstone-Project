## How to evaluate generative responses

#### Goals
- retireve relevant documents.
- Display some text relevant to users question.
  - Does the augmented answer improve the answer?
 
### Scoring

Do this manually. Finally use GPT-4 via copilot as the benchmark? Also maybe use credits or paste in text?

from - https://www.vellum.ai/blog/how-to-evaluate-your-rag-system

Answer Relevancy: How relevant is the answer to the question at hand?
For example, if you ask: “What are the ingredients in a peanut butter and jelly sandwich and how do you make it?" and the answer is "You need peanut butter for a peanut butter and jelly sandwich," this answer would have low relevancy. It only provides part of the needed ingredients and doesn't explain how to make the sandwich.‍

Faithfulness: How factually accurate is the answer given the context?
You can mark an answer as faithful if all the claims that are made in the answer can be inferred from the given context. This can be evaluated on a (0,1) scale, where 1 is high faithfulness

‍Correctness: How accurate is the answer against the ground truth data?

‍Semantic similarity: How closely does the answer match the context in terms of meaning
