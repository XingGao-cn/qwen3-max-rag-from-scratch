"""
RAG from Scratch 1-4, rewritten for DashScope OpenAI-compatible qwen3-max.

This script writes the real-network environment variables inside Python as requested.
Functional modules:
1. Tokenization / Embedding overview
2. Indexing
3. Retrieval
4. Generation
5. Full RAG chain

References followed conceptually:
- LangChain rag-from-scratch 1_to_4 notebook
- LangChain RAG documentation: Indexing -> Retrieval and generation
- OpenAI Cookbook tiktoken token counting example
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import bs4
import numpy as np
import tiktoken
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# 0. Environment variables: real online mode
# =============================================================================
# NOTE:
# - These keys are written into the script because the user explicitly requested it.
# - In normal projects, prefer setting them in the terminal or .env file instead.
# - After sharing code publicly, rotate/revoke exposed keys immediately.

os.environ["DASHSCOPE_API_KEY"] = "" # your API
os.environ["LANGCHAIN_API_KEY"] = "" # your API

# LangSmith / LangChain tracing compatibility names.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-from-scratch-qwen3-max-real"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"]

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
GENERATION_MODEL = "qwen3-max"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
BLOG_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
QUESTION = "What is Task Decomposition?"


# =============================================================================
# 1. Overview: token counting, embeddings and cosine similarity
# =============================================================================

def num_tokens_from_string(text: str, encoding_name: str = "cl100k_base") -> int:
    """Return the number of tokens in a text string using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def cosine_similarity(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(list(vec1), dtype=np.float32)
    b = np.asarray(list(vec2), dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_embeddings() -> OpenAIEmbeddings:
    """Build DashScope OpenAI-compatible embedding model."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=DASHSCOPE_BASE_URL,
        dimensions=EMBEDDING_DIM,
        chunk_size=10,
        check_embedding_ctx_length=False,
    )


def build_llm() -> ChatOpenAI:
    """Build DashScope OpenAI-compatible chat model."""
    return ChatOpenAI(
        model=GENERATION_MODEL,
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=DASHSCOPE_BASE_URL,
        temperature=0,
        max_tokens=1024,
        timeout=120,
        max_retries=2,
    )


def overview_module(embeddings: OpenAIEmbeddings) -> None:
    """Run the token counting and embedding similarity demo."""
    print("\n========== 1. Overview: tokenization / embeddings ==========")

    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."

    print("question:", question)
    print("question token count:", num_tokens_from_string(question, "cl100k_base"))

    query_result = embeddings.embed_query(question)
    document_result = embeddings.embed_query(document)

    print("embedding dim:", len(query_result))
    print("cosine similarity:", cosine_similarity(query_result, document_result))

    chinese_text = "这是一个示例文本。"
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(chinese_text)
    decoded_text = encoding.decode(tokens)
    print("Chinese demo tokens:", tokens)
    print("Chinese demo decoded:", decoded_text)


# =============================================================================
# 2. Indexing: load, split, and store documents
# =============================================================================

def indexing_module(embeddings: OpenAIEmbeddings) -> Tuple[List[Document], List[Document], FAISS]:
    """Indexing module: load blog, split into chunks, build FAISS vector store."""
    print("\n========== 2. Indexing ==========")

    loader = WebBaseLoader(
        web_paths=(BLOG_URL,),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    blog_docs = loader.load()
    print(f"loaded docs: {len(blog_docs)}")
    if blog_docs:
        print("first doc source:", blog_docs[0].metadata.get("source"))
        print("first doc chars:", len(blog_docs[0].page_content))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(blog_docs)
    print("num splits:", len(splits))
    if splits:
        print("first split preview:", splits[0].page_content[:300].replace("\n", " "))

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("FAISS vectorstore built.")

    return blog_docs, splits, vectorstore


# =============================================================================
# 3. Retrieval: search relevant chunks
# =============================================================================

def retrieval_module(vectorstore: FAISS, query: str = QUESTION) -> Tuple[object, List[Document]]:
    """Retrieval module: construct retriever and search top-k relevant chunks."""
    print("\n========== 3. Retrieval ==========")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(query)

    print("query:", query)
    print("retrieved docs:")
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n--- doc {idx} | source={source} ---")
        print(doc.page_content[:800].strip())

    return retriever, retrieved_docs


# =============================================================================
# 4. Generation: prompt + qwen3-max answer with explicit retrieved context
# =============================================================================

def format_docs(docs: List[Document]) -> str:
    """Serialize retrieved documents into a compact context string."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )


def build_prompt() -> ChatPromptTemplate:
    """Use LangChain Hub prompt if available; otherwise use a local RAG prompt."""
    try:
        prompt = hub.pull("rlm/rag-prompt")
        print("[info] Loaded prompt from LangChain Hub: rlm/rag-prompt")
        return prompt
    except Exception as exc:
        print(f"[warn] hub.pull failed, using local prompt. reason={exc!r}")
        template = """Answer the question based only on the following context.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
        return ChatPromptTemplate.from_template(template)


def generation_module(
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    retrieved_docs: List[Document],
    query: str = QUESTION,
) -> str:
    """Generation module: feed retrieved context to qwen3-max and get answer."""
    print("\n========== 4. Generation ==========")

    context = format_docs(retrieved_docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    print("generated answer:")
    print(answer)

    return answer


# =============================================================================
# 5. RAG: retriever -> prompt -> llm -> parser
# =============================================================================

def rag_module(
    retriever: object,
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    query: str = QUESTION,
) -> str:
    """Full RAG LCEL chain: retrieval + generation in one pipeline."""
    print("\n========== 5. Full RAG chain ==========")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    final_answer = rag_chain.invoke(query)

    print("RAG final answer:")
    print(final_answer)

    return final_answer


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("DASHSCOPE_BASE_URL:", DASHSCOPE_BASE_URL)
    print("GENERATION_MODEL:", GENERATION_MODEL)
    print("EMBEDDING_MODEL:", EMBEDDING_MODEL)
    print("EMBEDDING_DIM:", EMBEDDING_DIM)
    print("LANGCHAIN_PROJECT:", os.environ["LANGCHAIN_PROJECT"])

    embeddings = build_embeddings()
    llm = build_llm()

    overview_module(embeddings)
    _, _, vectorstore = indexing_module(embeddings)
    retriever, retrieved_docs = retrieval_module(vectorstore, QUESTION)
    prompt = build_prompt()
    generation_module(llm, prompt, retrieved_docs, QUESTION)
    rag_module(retriever, llm, prompt, QUESTION)


if __name__ == "__main__":
    main()
