import gradio as gr
import sqlite3
from sentence_transformers import SentenceTransformer, util
import numpy as np
from gpt4all import GPT4All
import cohere
from config import Keys
from openai import OpenAI


## to do
## Add selection box for model
## make clear screen button and user-feedback
## add default question so can't be empty


DB = "12-4-24-all-mini-token-100.db"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
gen_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
co = cohere.Client(Keys.cohere)
gen_model_name = "orca-mini"
OPENAI_API_KEY=Keys.openAI
client = OpenAI(api_key=OPENAI_API_KEY)

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
    default_ans = None
    cand = candidates.split("\n\n")

    prompt_temp = f"""The original query is as follows: {question}
                We have provided an existing answer: {default_ans}

                We have the opportunity to refine the existing answer (only if needed) with some more context below.
                ------------

                {cand[0]}

                ------------
                Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.

                Refined Answer:
                """
    # if there are no additional context just return 
    if len(candidates) == 0:
            prompt_temp = question

    default = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are an AI assistant, you are skilled at answering queries for people writing literature reviews."},
        {"role": "user", "content": question}
    ],
    max_tokens=300
    )
    default_ans = default.choices[0].message.content

    ## Incorporate docs
    context_ans = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are an AI assistant, you are skilled at answering queries for people writing literature reviews."},
        {"role": "user", "content": prompt_temp}
    ],
    max_tokens=300
    
    )

    return context_ans.choices[0].message.content

        #orca-mini
        # default_ans = gen_model.generate(prompt=question, temp=0)

        # if len(candidates) == 0:
        #     prompt_temp = question
        # #print(prompt_temp)
        # #ans = gen_model.generate_text(prompt=prompt_temp, temp=0)
        # # gpt4all
        # ans = gen_model.generate(prompt=prompt_temp, temp=0)
        # #print(ans)
        # return ans

    

def feedback(rd, rgt, at, fg):
    # fix this then add database and done....
    print(rd, rgt, at, fg)

with gr.Blocks() as demo:
    demo.title = "WRAP Doc Chat"
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Ask a Question", placeholder="How should I store Bananas?")
            submit_q_btn = gr.Button(value="Submit")
        with gr.Column():
            retrieved = gr.Textbox(label="Retrieved text")
            from_docs = gr.Textbox(label="Relevant documents")
    
    with gr.Tabs():
        with gr.TabItem("Generate Content"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        generate_btn = gr.Button(value=f"Generate answer")
                with gr.Column():
                    answer = gr.Textbox(label="Generated text")
        with gr.TabItem("Feedback"):
            with gr.Row():
                gr.Textbox(value="here is how to scoring works")
            with gr.Row():
                rd = gr.Slider(1,10,5,label="Relevance of documents", interactive=True)
            with gr.Row():
                rgt = gr.Slider(1,10,5,label="Relevance of generated text", interactive=True)
            with gr.Row():
                at = gr.Slider(1,10,5,label="Adherance to text", interactive=True)
            with gr.Row():
                fg = gr.Slider(1,10,5,label="Faithfullnes of generated text", interactive=True)
            with gr.Column():
                    submit_fbk_btn = gr.Button(value="Submit Feedback")


    # button to search db
    submit_q_btn.click(meta, inputs = question, outputs=[from_docs, retrieved])
    # generate answer
    generate_btn.click(generate, inputs = [question, retrieved], outputs=[answer])
    # submit feedback
    submit_fbk_btn.click(feedback)
        
        
demo.launch(auth=("", ""))
