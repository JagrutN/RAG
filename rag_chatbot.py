import logging
import re
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal



currentDir = os.getcwd()
logDir = os.path.join(currentDir,"logs")
class PubMedBertEmbeddings:
    def __init__(self, model, tokenizer, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def createEmbeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text):
        return self.createEmbeddings(text)  # ChromaDB expects a single vector

    def embed_documents(self, texts):
        return self.createEmbeddings(texts)  # ChromaDB expects a list of vectors

class RAG:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)  # Logger name is the class name
        self.logger.setLevel(logging.DEBUG)

        output_handler = logging.FileHandler(os.path.join(logDir, "output.log"))
        output_handler.setLevel(logging.INFO)  # Logs INFO and above

        error_handler = logging.FileHandler(os.path.join(logDir, "error.txt"))
        error_handler.setLevel(logging.ERROR)  # Logs only ERROR and above

        #Create Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Attach formatter to handlers
        output_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)

        # Attach handlers to logger
        self.logger.addHandler(output_handler)
        self.logger.addHandler(error_handler)

        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.chat_model_name = "openchat/openchat-3.5-1210"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self.max_tokens = 512
        self.temperature = 0.7
        self.repetition_penalty = 1.1
        self.k = 3
        self.chat_model = None
        self.retriever = None
        self.rag_chain = None
        self.vectorDB = None

    def parseData(self):
        '''
        Loadig files and returning as document class.
        Utilizing Langchain TextLoader
        '''
        dataDir = os.path.join(currentDir, "data")
        textFiles = [file for file in os.listdir(dataDir) if file.endswith(".txt")]
        docs = []

        for file in textFiles:
            filePath = os.path.join(dataDir, file)
            try:
                self.logger.info(f"Prosessing file {file}")
                loader = TextLoader(filePath)
                document = loader.load()
                docs.extend(document)
                self.logger.info(f"Successfully Parsed file {file}")

            except Exception as e:
                self.logger.error(f"Error in file {file}")
                self.logger.error(f"Error : {e}")

        self.logger.info(f"Successfully fineshed loading {len(docs)} documents")
        return docs
    
    def splitDocuments(self, docs, chunkSize, chunkOverlap):
        '''
        Splitting documnets and creating chunks.
        Chunks created based on chunkSize and chunkOverlap
        '''

        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size = chunkSize,
            chunk_overlap = chunkOverlap
        )
        chunks = []
        for doc in docs:
            try:
                self.logger.info(f"Chunking document {doc.metadata['source']}")
                chunk = textSplitter.split_documents([doc])
                chunks.extend(chunk)
                self.logger.info(f"Succesfully chunked document {doc.metadata['source']}")
            except Exception as e:
                self.logger.error(f"Error in chunking document {doc.metadata['source']}")
                self.logger.error(f"Error : {e}")

        self.logger.info(f"Successfully created {len(chunks)} chunks")
        return chunks
    
    def createEmbeddings(self, text):
        '''
        Converts text to Embeddings using PubMedBert
        '''
        inputs = self.tokenizer(text, return_tensors= 'pt', padding= True, truncation= True, max_length= 512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings.tolist()
    
    def createEmbeddingsandVectorStore(self, chunks):
        '''
        Parse chunk one by one and create Embeddings and Vector Store
        '''
        embeddingFunction = PubMedBertEmbeddings(model= self.model, tokenizer= self.tokenizer, device= self.device)
        self.vectorDB = Chroma(
            collection_name= "medical_docs",
            persist_directory= "./chroma_db",
            embedding_function= embeddingFunction)
        
        self.retriever = self.vectorDB.as_retriever(search_kwargs={"k": self.k})
        
        embeddings = []
        documents = []
        metadataList = []
        ids = []

        for i, chunk in enumerate(chunks):
            try:
                self.logger.info(f"Creating Embeddings for {i} chunk of {chunk.metadata['source']}")
                embedding = self.createEmbeddings(chunk.page_content)

                docId = f"doc_{i}"
                ids.append(docId)
                embeddings.append(embedding)
                documents.append(chunk.page_content)
                metadataList.append(chunk.metadata)

            except Exception as e:
                self.logger.error(f"Error in Creating Embedding for {i} chunk of {chunk.metadata['source']}")
                self.logger.error(f"Error : {e}")

        try:
            self.logger.info("Storing in Vector DB")

            self.vectorDB._collection.add(
                documents = documents,
                embeddings = embeddings,
                metadatas = metadataList,
                ids = ids
            )

            self.vectorDB.persist()

            self.logger.info("Successfully stored all the Embeddings and Page content in Vector Store")
        except Exception as e:
            self.logger.error("Error inserting documents in vectorDB")
            self.logger.error(f"Error : {e}")

    def initializeChatModel(self):
        '''
        Initialize the chat model Hugging Face 
        Model Utilized = meta-llama/Llama-3.2-1B-Instruct
        '''

        try:
            self.logger.info("Initializing the chat model")

            chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_name)
            chat_model = AutoModelForCausalLM.from_pretrained(
                self.chat_model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )

            # Define the pipeline
            pipe = pipeline(
                task="text-generation",
                model=chat_model,
                tokenizer=chat_tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                repetition_penalty=self.repetition_penalty,
            )

            pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

            llm = HuggingFacePipeline(pipeline = pipe)
            self.chat_model = ChatHuggingFace(llm=llm)

            self.logger.info(f"Initialized the chat model {self.chat_model_name}")
            print(f"Initiaized the chat model {self.chat_model_name}")

        except Exception as e:
            self.logger.error(f"Falied initialization of chat model {self.chat_model_name}")
            self.logger.error(f"Error: {e}")
            print(f"Falied initialization of chat model {self.chat_model_name}")
            print(f"Error: {e}")
            sys.exit()

    def createRagChain(self):

        try:
            self.logger.info("Starting creating Rag Chain")
            promptTemplate = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            """

            prompt = PromptTemplate.from_template(promptTemplate)

            if self.retriever is None:
                raise ValueError("Retriever is not initialized. Run `createEmbeddingsandVectorStore()` first.")
            

            def format_docs(docs):
                if not docs:
                    return "No relevant documents found"
                return "\n\n".join([doc.page_content for doc in docs])
            
            def cleanup_text(text: str) -> str:
                text = text.content.replace("Answer: ", "")
                text = re.sub(r"<\|.*?\|>", "", text)
                return text
            
            cleanup_runnable = RunnableLambda(cleanup_text)
            
            self.rag_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | self.chat_model
                | cleanup_runnable
                | StrOutputParser()
            )
            self.logger.info("Rag Chain Created")
        except Exception as e:
            self.logger.error("Error Creating Rag Chain")
            self.logger.error(f"Error: {e}")

    def create_agentic_rag_graph(self):
        class State(TypedDict):
            query: str
            result: str

        def router(state: State) -> Literal["llm", "retriever", "rag"]:
            query = state["query"]

            routing_prompt = f"""
                You are a routing agent in a medical assistant system.

                Given the user query below, decide which path to take:

                - Return "retriever" if the user is asking to **search for or find information from stored documents** (e.g., "Find patient records mentioning diabetes").
                - Return "llm" if the user has:
                    - If the user wants an explanation or summary, OR
                    - Asked a general knowledge question that does not require document retrieval (e.g., "What is the capital of India?", "What is diabetes?", "How are you")
                - Return "rag" if the query requires both document retrieval and LLM reasoning (e.g., "Summarize findings from uploaded lab results").

                Respond ONLY with one of the following: retriever, llm, rag.

                I just want one word answer.

                Query:
                {query}
                """

            try:
                response = self.chat_model.invoke(routing_prompt).content.strip().lower()
                text = re.sub(r"<\|.*?\|>", "", response).strip().lower()

        # Extract from model response
                match = re.search(r"gpt4 correct assistant:\s*(\w+)", text)
                route = match.group(1) if match else text.split()[-1] if text else "rag"
                if route not in ["retriever", "llm", "rag"]:
                    self.logger.warning(f"Invalid route: {route}. Defaulting to 'rag'")
                    return {"next": "rag"}
                return {"next": route}
            except Exception as e:
                self.logger.error(f"Routing failed: {e}")
                return {"next": "rag"}
            
        def retriever_node(state: State):
            docs = self.retriever.get_relevant_documents(state["query"])
            return {"result": "\n\n".join([doc.page_content for doc in docs])}

        def llm_node(state: State):
            return {"result": self.chat_model.invoke(state["query"]).content}

        def rag_node(state: State):
            return {"query": state["query"], "result": self.rag_chain.invoke(state["query"])}

        builder = StateGraph(State)

        builder.add_node("router", router)
        builder.add_node("retriever", retriever_node)
        builder.add_node("llm", llm_node)
        builder.add_node("rag", rag_node)

        builder.set_entry_point("router")
        builder.add_conditional_edges("router", lambda state: state["next"], {
            "llm": "llm",
            "retriever": "retriever",
            "rag": "rag",
        })

        builder.add_edge("llm", END)
        builder.add_edge("retriever", END)
        builder.add_edge("rag", END)

        self.agentic_rag_graph = builder.compile()
        self.logger.info("Agenti graph created")