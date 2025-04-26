from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os


load_dotenv()
model_name = os.environ.get('LLM_MODEL')


class GradeDocuments(BaseModel):
        """ Binary score for relevence check for retrived documents"""

        binary_score: str =Field(
            description="Documents are relevnt to the question, 'yes' or 'no' "
        )

chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=0)




# Apply structured output to the model
structured_llm_grader = chat_model.with_structured_output(GradeDocuments)

system= """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader