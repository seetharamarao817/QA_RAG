from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.chat_models.openai import ChatOpenAI
import pprint
import json
import operator
from typing import Sequence, TypedDict, Dict
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
import os
from langchain.chains.summarize import load_summarize_chain




def grade_question(state):
    """
    Determines whether the question is a general question or summarization question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The type of question
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""System: ### Instructions:
    You are a binary classifier. Your task is to determine if a user question requires summarizing the ENTIRE document to answer it.
    **Criteria for 'yes' (entire document summarization):**
    * The question is about summarizing information.
    * Answering the question requires considering the whole document.

    **Examples:**

    **Yes:**
    * "Can you summarize the main agendas of this city council meetings ?"
    * "What are the key events described in this city council meetings?"

    **No:**
    * "What is the bid amount of Calico California Constructores, Inc." (Factual information)
    * "Explain about policy lu 1-4 in table 1 general plan consistency" (Specific information)
    User question: Here is the user question: {question} \n
    Assistant:
    Give a binary score 'yes' or 'no' score to indicate whether the question is entire document summarization or not. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["question"],
    )

    # Chain
    chain = prompt | llm | JsonOutputParser()

    # Score
  
    score = chain.invoke({
                "question": question}
           )

    return {"keys": {"question": question,"score":score,"retriever":state_dict["retriever"],"chunks":state_dict["chunks"]}}

def decide_to_generate(state):
    """
    Determines whether to summarize or answer the question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    score = state_dict["score"]

    if score["score"] == "yes":
        print("---DECISION: Summarization QUERY---")
        return "summary_query"
    else:
        print("---DECISION: General QUERY---")
        return "general_query"

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents= state_dict["retriever"].get_relevant_documents(question)
    return {"keys": {"question": question,"documents":documents,"retriever":state_dict["retriever"],"chunks":state_dict["chunks"]}}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"question": question, "generation": generation}
    }


def summarize(state):

    state_dict = state["keys"]
    question = state_dict["question"]
    map_template = """The following is a set of documents:
{text}
Based on these documents please write a concise summary.
The summary should be accurate, relevant, and concise. It should avoid redundancy and also faithfully represent the key points from the documents. 
Helpful Answer:"""
  
    map_prompt = PromptTemplate(template=map_template,input_variables=['text'])
  
    reduce_template = """The following is a set of summaries :
{text}
Take these summaries and distill them into a final, consolidated summary. The final summary should be accurate, relevant, and concise.

Helpful Answer:
  """
    reduce_prompt = PromptTemplate(template=reduce_template,input_variables = ['text'])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    map_reduce_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=reduce_prompt,
    return_intermediate_steps=False,
)
    
    text_chunks = state_dict["chunks"]
    map_reduce_outputs = map_reduce_chain.invoke(input=text_chunks)
    generation = map_reduce_outputs['output_text']
    return {
      "keys": {"question": question, "generation": generation}
    }

def prepare_for_final(state):
    """
    Passthrough state for final grade.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The current graph state
    """

    print("---FINAL GRADE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    generation = state_dict["generation"]

    return {
        "keys": {"question": question, "generation": generation}
    }


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

def summary_workflow():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_question", grade_question) 
    workflow.add_node("generate", generate)  
    workflow.add_node("summarize", summarize)  
    workflow.add_node("prepare_for_final", prepare_for_final)  # passthrough


    # Build graph
    workflow.set_entry_point("grade_question")
    workflow.add_conditional_edges(
    "grade_question",
    decide_to_generate,
    {
    "general_query": "retrieve",
    "summary_query": "summarize",
    },
    )
    workflow.add_edge("retrieve","generate")
    workflow.add_edge("generate","prepare_for_final")
    workflow.add_edge("summarize","prepare_for_final")
    workflow.add_edge("prepare_for_final",END)


    return workflow.compile()


