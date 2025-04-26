import os
from dotenv import load_dotenv
import pandas as pd 
import bs4

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chonkie import SemanticChunker
from markitdown import MarkItDown



load_dotenv()

model_name = os.environ.get('LLM_MODEL')
data_path = os.environ.get('DATA_PATH')
db_location=os.environ.get('CHROMA_DB')
db_exists = os.path.exists(db_location)


def process_stock_data(stock_data_dir):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        stock_data: List of Documents.
    """
    stock_data = []
    for file in os.listdir(stock_data_dir):
        print(f"Processing file: {file}")
        file_path = os.path.join(stock_data_dir, file)
        df= pd.read_csv(file_path)
        df['text']= df.apply(lambda row: f"Stock {row['stock']} on date {row['date']}, opening price {row['ouverture']:.2f}, closing price {row['cloture']:.2f}, volume {row['volume']:,.2f}.", axis=1)
        stock_data += df['text'].tolist()

    stock_data = "\n".join(stock_data)
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=100,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
    splits = text_splitter.split_text(stock_data)
 
    return splits


def process_pdf(file_path):
    """
    Load data from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        list: Loaded data as a list of IDs.
    """
    md = MarkItDown()
    result = md.convert(file_path)
    chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
    chunks = chunker.chunk(result.text_content)
    all_splits = [Document(page_content=chunk.text) for chunk in chunks]
    try:
        ids = stock_vector_store.add_documents(all_splits)
    except Exception as e:
        print(f"Error adding documents: {e}")  
        ids = []
    return ids      
    

def process_urls(urls) -> list[Document]:
    """
    Load data from a list of URLs.
    
    Args:
        urls (Sequence): Sequence of URLs to load data from.
    
    Returns:
        splits: List of documents.
        
    """
    bs4_strainer = bs4.SoupStrainer(class_=("h1n mob30", "inarticle txtbig"))
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
    splits = text_splitter.split_documents(docs)

    return splits



embedding_model = GoogleGenerativeAIEmbeddings(model=model_name) 


urls = ("https://www.ilboursa.com/marches/le-directeur-general-de-banque-zitouna-nabil-el-madani-limoge_52322",
        "https://www.ilboursa.com/marches/la-turquie-s-engage-dans-l-exploration-petroliere-et-gaziere-en-libye_52271",
        "https://www.ilboursa.com/marches/kilani-holding-ne-prevoit-pas-de-retirer-la-sta-de-la-cote-de-la-bourse_52315")

if db_exists:
    stock_vector_store = Chroma(
        collection_name="stock_collection",
        persist_directory=db_location,
        embedding_function=embedding_model,
    )
    
else:
    stock_vector_store = Chroma(
            collection_name="stock_collection",
            persist_directory=db_location,
            embedding_function=embedding_model,
        )
    _ =stock_vector_store.add_texts(process_stock_data(data_path))
    _ =stock_vector_store.add_documents(process_urls(urls))

    