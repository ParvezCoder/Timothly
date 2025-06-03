import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ----- Streamlit Page Config -----
st.set_page_config(page_title="AI Project for Timothly", page_icon="🤖", layout="centered")

# ----- Custom Dark Theme CSS -----
custom_css = """
<style>
body {
    background-color: #0f1117;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background-color: #0f1117;
}
header, footer {
    visibility: hidden;
}
div[data-testid="stTextInput"] > div > input {
    background-color: #1e1f26;
    color: white;
    border: 1px solid #333;
}
textarea, .stTextInput input {
    background-color: #1e1f26 !important;
    color: #fff !important;
}
.stButton button {
    background-color: #6c63ff;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #837dff;
}
.code-style {
    background-color: #1e1f26;
    padding: 15px;
    border-radius: 10px;
    font-family: monospace;
    color: #ffffff;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----- Title -----
st.markdown("<h1 style='text-align: center; color: #6c63ff;'>🔍 AI Project for Timothly</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #ccc;'>Ask anything — answers are generated from curated documents and live website data.</p>", unsafe_allow_html=True)

# ----- Website Scraper -----
def scrape_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        texts = []
        for tag in ['p', 'div', 'span', 'li', 'h1', 'h2', 'h3']:
            for element in soup.find_all(tag):
                text = element.get_text(strip=True)
                if text:
                    texts.append(text)
        return ' '.join(texts)
    except Exception as e:
        return f"Error scraping website: {str(e)}"

# ----- Document Setup -----
base_documents = [
    Document(page_content="""Elon Musk is a billionaire entrepreneur, inventor, and engineer known for 
             founding and leading several groundbreaking companies. He is the CEO of Tesla, which makes electric
              vehicles; SpaceX, which develops rockets and spacecraft; and Neuralink, which works on brain-computer 
             interfaces. He also co-founded PayPal and is the owner of X (formerly Twitter). """, metadata={"source": "newsletter"}),
    Document("""Name: Timothy
                Conatct no: +61 459 469397
                Website : https://intergriai.co.site
                CEO of: IntergriAI Solution
                Monthly earning = $100,000
                email_id = Timothly@intergriai.com
                """, metadata={"source": "https://intergriai.co.site/"}),
    Document(page_content="Goldfish are popular pets", metadata={"source": "jang"}),
    
    Document(page_content="""
                Name = Engr. Parvez Ahmed
                CEO of : = ReXon Solution
                website: = https://aicoderr.vercel.app/
                ADDRESS:
                H: 177, Gulistan Society, Near Dost M G. Store, Landhi, Karachi.
                PERSONAL INFO:
                • Email: ParvezCoder786@gmail.com
                • Contact Number: +92 305 288 7779
                • Website: https://aicoderr.vercel.app
                • LinkedIn: linkedin.com/in/parvez-ahmed-1604b92b5
                • GitHub: github.com/ParvezCoder
                EDUCATION:
                • B.E in Computer System Engineering, QUEST
                EXPERIENCE:
                • Ornesol Pvt Ltd - AI/ML Chatbot Expert (2024 - Present)
                SKILLS:
                • Python, AI, ML, Deep Learning, React.js, Next.js, Tailwind, MySQL
                PROJECTS:
                • Portfolio Website, Amazon Clone, Virtual Assistant
                """, metadata={"source": "https://aicoderr.vercel.app/"}),

]

# Scrape IntergriAI website
web_url = "https://intergriai.co.site/"
web_text = scrape_website_text(web_url)
web_doc = Document(page_content=web_text, metadata={"source": web_url})

# Combine documents
documents = base_documents + [web_doc]

# ----- Embedding & Vector Store -----
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

vectorstore = FAISS.from_documents(documents, embedding=embeddings)

# ----- LLM -----
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv('GOOGLE_API_KEY'),
)

# ----- Prompt -----
# -------- PROMPTS --------
rag_prompt = ChatPromptTemplate.from_messages([
    ("human", "Answer this question using only the provided content:\n\nQuestion: {question}\n\nContent:\n{content}")
])
fallback_prompt = ChatPromptTemplate.from_messages([
    ("human", "You are a smart AI assistant. Answer the question as best as you can using your own knowledge.\n\nQuestion: {question}")
])

# ----- Helper: Retrieve or Fallback with Score -----
def get_relevant_docs_or_fallback(question, score_threshold=0.5):
    results = vectorstore.similarity_search_with_score(question, k=1)
    if not results:
        return None
    doc, score = results[0]

    # Log similarity info
    print("----- Similarity Check -----")
    print(f"User Question: {question}")
    print(f"Top Document Content: {doc.page_content[:100]}...")
    print(f"Similarity Score: {score}")

    if score < score_threshold:
        return None
    return doc

# ----- Streamlit Input -----
user_question = st.text_input("Enter your question:", placeholder="e.g. What is IntergriAI solution?")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                relevant_doc = get_relevant_docs_or_fallback(user_question)
                if relevant_doc:
                    content = relevant_doc.page_content
                    formatted_prompt = rag_prompt.format(question=user_question, content=content)
                else:
                    formatted_prompt = fallback_prompt.format(question=user_question)

                response = llm.invoke(formatted_prompt)
                st.markdown("<div class='code-style'>" + response.content + "</div>", unsafe_allow_html=True)

                if not relevant_doc:
                    st.info("Note: This is a general AI-generated answer (not from your documents).")

            except Exception as e:
                st.error(f"Error: {str(e)}")
