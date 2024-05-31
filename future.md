## Lessons learned - what I would do next time

- Create synthetic questions - hundreds... from document chunks

- Track metrics for each query e.g. cosine distance in order to track difficult questions.

- Do full text search as well appraently will [improve system](https://jxnl.co/writing/2024/05/11/low-hanging-fruit-for-rag-search/#4-tracking-average-cosine-distance-and-cohere-reranking-score)

- Add meta data from docuements (have seen other systems do this e.g. H20GPT)

- https://docs.parea.ai/tutorials/getting-started-rag (somethig to read for optimising pipelines.

- The ability for searching based on time (e.g what is the most recent...?)

## Extracting data from PDFs
It is quite challlenging and could easily be a project on its own. In future get the text before it is presented in a pdf and avoid having to process a pdf lots of info on extraction from pdfs [here](https://unstract.com/blog/pdf-hell-and-practical-rag-applications/)

## Document search

- classical methods of searching documents might be more effective in some situations - especially when the query is not in the form of the question. Therefore having a stack that uses classical methods alongside vector search would be optimal.

  If i were doing this project over again, I would only focus on this bit and then just pass the text to openAI for the final bit. Finding the data in the first place is the most important bit.
