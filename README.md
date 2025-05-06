# Adaptive_Rag_TN_Stock_Market

This project implements an **Adaptive RAG (Retrieval-Augmented Generation)** system to assist users in making **stock market investment decisions** based on real-time information and relevant documents. The system uses **LangChain**, **LangGraph**, and **Pinecone** for vector storage, providing an intelligent, self-reflecting agent capable of improving the results through query refinement and web search integration.

## Features

- **Adaptive RAG**: The system dynamically processes user queries and retrieves relevant documents from a stock market knowledge base.
- **Self-Reflection**: The system automatically rewrites the query if the retrieved documents are irrelevant, ensuring more accurate results.
- **Web Search Integration**: When necessary, the system can query web sources for additional stock market data, improving the quality of responses.
- **Streamlit Web Interface**: The user interface is built using **Streamlit**, making it easy to interact with the system through a browser.

## Architecture

The system is designed based on the **Adaptive RAG** architecture, which utilizes multiple nodes for document retrieval, relevance grading, and self-reflection. The architecture supports refining the question based on relevance feedback and allows for web searches when documents are insufficient.

### Adaptive RAG System Architecture

The following diagram represents the architecture of the **Adaptive RAG** system:

![Adaptive RAG System Architecture](https://github.com/user-attachments/assets/1a206585-95d7-45b0-8de9-884a609bc68e)


> The image above is sourced from the [LangChain Adaptive RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/).

### System Workflow:

1. **Query Analysis**: The system first analyzes the question to determine if it is related to the available index of documents. If it is unrelated, a web search is performed for additional information.
2. **Document Retrieval**: Relevant documents are retrieved from a vector store (ChromaDB), based on the user's query. The vector store contains information about the stock trends, and latest market news.
3. **Grading and Relevance Check**: The relevance of the retrieved documents is graded, and if the documents are deemed irrelevant, the system rewrites the question to improve the search results.
4. **Generate Response**: A generative model (Gemini in this project) is used to generate an answer based on the relevant documents and the query.
5. **Self-Reflection**: If the model’s generated response is not satisfactory or the documents seem to be irrelevant, the system rewrites the question and refines the process to get better results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedrezgui/Adaptive_Rag_TN_Stock_Market.git
   ```

2. Create a Virtual Environment and install Dependencies:
```bash
conda env create -f environment.yml

conda activate Adaptive_Rag
```

## Usage

1. **Run the Streamlit Web Interface**:
   After setting up the project, navigate to the project directory and start the Streamlit application:
   ```bash
   streamlit run src/main.py
   ```

2. **Interact with the System**:
   Open your browser and go to `http://localhost:8501` to interact with the system. Enter stock-related questions, and the system will fetch relevant documents and generate answers based on its RAG system.


## Example

![image](https://github.com/user-attachments/assets/d07e57b6-37ab-407f-8c0c-bee39c6d2b77)


## Development

- **Adding More Data**: You can add more stock-related documents to improve the system's knowledge base by uploading them directly from the web interface in the sidebar.
- **Improving the Query Generation**: You can improve the query generation logic by fine-tuning the language model on a stock-specific dataset or using other generative models.

## Contributing

Contributions are welcome! Feel free to fork this repository, submit issues, and send pull requests to improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Conclusion

This **Adaptive RAG** system provides an intelligent way to help users make investment decisions based on real-time information and relevant document retrieval. With **LangChain**, **LangGraph**, **ChromaDB**, and **Streamlit**, this project offers a powerful framework for building advanced, dynamic question-answering systems.
