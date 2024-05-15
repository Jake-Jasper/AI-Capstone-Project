import gradio as gr
import sqlite3
from sentence_transformers import SentenceTransformer, util
import numpy as np
from gpt4all import GPT4All
import cohere
from config import Keys


DB = "12-4-24-all-mini-token-100.db"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
gen_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
co = cohere.Client(Keys.cohere)
gen_model_name = "orca-mini"

# find best docs
def find_top_k_relevance(question=["No question"], model=None, n=20, db=DB):
  q = model.encode(question)
  scores = {}
  conn = sqlite3.connect(db)

# Create a cursor object
  cursor = conn.cursor()
  # load row by row
  cursor.execute('SELECT * FROM documents')
  for row in cursor:
    scores[row[0]] = util.pytorch_cos_sim(q, np.frombuffer(row[4], dtype=np.float32)).numpy()

  return dict(sorted(scores.items(), key = lambda x: x[1], reverse = True)[:n])

# get docs and re rank
def get_relevant_docs(top_k, q):
    top_docs = list(top_k.keys())

    conn = sqlite3.connect(DB)
    docs = []
    # Create a cursor object
    cursor = conn.cursor()
    # load row by row
    # have to do the comprehension to unpakc inside and f-string
    # this doesn't get then in order
    for i in top_docs:
        docs.append(cursor.execute(f"""SELECT sentence FROM documents
                    WHERE id = {i}""").fetchone())

    docs = [i[0] for i in docs]
    results = co.rerank(model = "rerank-english-v3.0", query=q, documents=docs, top_n=3, return_documents=True)
    relevant_ids = []
    # relevant doc ids
    rel_documents = []
    # texts for prompt
    prompt_candidates = []

    for idx, r in enumerate(results.results):
        # only get the best ones..cohere seems to rank them higher than with cosine so 0.4 should be good
        if r.relevance_score > 0.7:
            relevant_ids.append(r.index)

    db_ids = [top_docs[i] for i in relevant_ids]

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    for i in db_ids:
        x = cursor.execute(f"""SELECT * FROM documents
                        WHERE id = {i}""").fetchone()

        rel_documents.append(x[2])
        prompt_candidates.append(x[3])

    conn.close()

    if len(set(rel_documents)) == 0:
        return "No relevant documents found" , ""
    else:
        return set(rel_documents), prompt_candidates

def meta(question):
    top_k = find_top_k_relevance(question=question, model=model)
    docs, candidates = get_relevant_docs(top_k, question)
    newline = "\n\n"
    # format returned strings
    if docs == "No relevant documents found":
        return docs, candidates
    else:    
        candidates = newline.join(f"{candidate}" for candidate in candidates)
        docs = newline.join(f"{doc}" for doc in docs)

    return docs, candidates


def generate(question, candidates):
    # some reason it gives blank text if I pass in all data...
    cand = candidates.split("\n\n")

    prompt_temp = f"""### System:
        You are an AI Assistant that helps to answer questions as best you can and incorporate user input.
        ### User:
        {question}
        
        ### System:
        Based on the retrieved information from the pdf, here are the relevant excerpts:
        
        {cand[0]}
        
        Please answer the user's question, integrating insights from these excerpts and your general knowledge. Limit your answer to Three sentences in the form of a paragraph.
        ### Response:
        """
    if len(candidates) == 0:
        prompt_temp = question
    #print(prompt_temp)
    #ans = gen_model.generate_text(prompt=prompt_temp, temp=0)
    # gpt4all
    ans = gen_model.generate(prompt=prompt_temp, temp=0)
    #print(ans)
    return ans


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Ask a Question")
            submit_q_btn = gr.Button(value="Submit")
        with gr.Column():
            retrieved = gr.Textbox(label="Retrieved text")
            from_docs = gr.Textbox(label="Relevant documents")
    # generate answer
    with gr.Row():
        with gr.Column():
            generate_btn = gr.Button(value=f"Generate question answer using {gen_model_name}")
        with gr.Column():
            answer = gr.Textbox(label="Generated text")
    ## make feedback here


    # button to search db
    submit_q_btn.click(meta, inputs = question, outputs=[from_docs, retrieved])
    # button to generate text
    generate_btn.click(generate, inputs = [question, retrieved], outputs=[answer])
    
    
demo.launch()
