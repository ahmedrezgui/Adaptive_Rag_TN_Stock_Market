from typing import List, TypedDict
from langgraph.graph import StateGraph, END,  START
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
from utils import *
import os


load_dotenv()


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    documents: List[Document]= []
    generation :str =""


def retrieve(state):
    """
    Retrieve Documents

    Args:
        state (State): The state of the graph.

    Returns:
        state(dict): The state of the graph with the retrieved documents in a new key.    
    """
    question= state["question"]
    documents = stock_vector_store.similarity_search(
        query=question,
        k=20,
        )
    with open('output.txt', 'w') as file:
        for item in documents:
            file.write(item.page_content + '\n')
    return {"documents": documents ,  "question": question }

def grade_documents(state: State):

        print("---CHECK DOCUMENT RELEVNECE TO QUESTION ---")
        question = state['question']
        documents = state['documents']

        filtered_docs=[]
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            print(score)
            grade= score.binary_score
            if grade == "yes":
                print(f"Document is relevant to the question")
                filtered_docs.append(d)
            else:
                print(f"Document is not relevant to the question") 
                continue
        return {"documents": filtered_docs, "question": question }

def transform_query(state:State):

        print("rewriting question")
        better_question = question_rewriter.invoke({"question": state["question"]})
        return ({"question": better_question})

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def generate(state:State):
        """
        Generate a response based on the question and documents.

        Args:
            state (State): The state of the graph.

        Returns:
            state (dict): The state of the graph with the generated response in a new key.
        """
        question = state["question"]
        documents = state["documents"]
        top_contexts = [doc.page_content for doc in documents]
        generation = generation_chain.invoke({"question": question, "context": top_contexts})
        return {"generation": generation, "question": question , "documents": documents }

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader_agent.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        # print("---GRADE GENERATION vs QUESTION---")
        # score = answer_grader_agent.invoke({"question": question, "generation": generation})
        # grade = score.binary_score
        # if grade == "yes":
        #     print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
        # else:
        #     print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        #     return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def create_workflow():

    workflow = StateGraph(State)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search) 
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)

    workflow.add_conditional_edges(
        START,
        route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            }   
        )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: ("generate" if len(state["documents"]) > 0 else "transform_query"),
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }   
    )
    workflow.add_conditional_edges(
        "transform_query",
        route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            }   
        )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
             "useful": END,
            #  "not useful": "transform_query",   
             "not supported": "transform_query",
             
        }
    )

    app = workflow.compile()

    return app
