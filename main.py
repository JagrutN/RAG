import sys
import argparse
import logging
import numpy as np
import torch
from rag_chatbot import RAG

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunk_size", type=int, help= "Maximum size of text to be passed to embedding model")
    parser.add_argument("--chunk_overlap", type= int, help = "Size of text to overlap between two strings after chunking")

    args = parser.parse_args()

    chunkSize = args.chunk_size
    chunkOverlap = args.chunk_overlap

    rag = RAG()

    logger = logging.getLogger("RAG")
    logger.info("Initializing RAG Chatbot")

    docs = rag.parseData()
    chunks = rag.splitDocuments(docs, chunkSize= chunkSize, chunkOverlap= chunkOverlap)
    rag.createEmbeddingsandVectorStore(chunks)

    rag.initializeChatModel()
    rag.createRagChain()

    print("Your Rag Chatbot is created")

    try:
        logger.info("Welcome User you will be intergacting with a Medical ChatBot")
        print("Welcome User you will be intergacting with a Medical ChatBot")
        choice = True
        while choice:
            question = input("Enter your Question: ")
            logger.info(f"Question: {question}")
            print(f"Question: {question}")

            retrieved_docs = rag.retriever.get_relevant_documents(question)

            # Print retrieved documents for debugging
            print("\n=== Retrieved Documents ===")
            for idx, doc in enumerate(retrieved_docs):
                print(f"Doc {idx+1}: {doc}\n")
            print("===========================\n")

            answer = rag.rag_chain.invoke(question)
            answer = answer.replace("Answer: ", "")
            print(f"Answer: {answer}")
            logger.info(f"Answer: {answer}")
            choice = input("Do you want to ask more questions? [Y/ N/ No/ Yes]")
            choice = choice.lower()
            if choice in ["n", "no"]:
                logger.info("Choice: No")
                logger.info("Exiting!")
                logger.info("-------------------------------------")

                print("Choice: No")
                print("Exiting the chatbot!")
                print("-------------------------------------")
                choice = False
            elif choice in ["y", "yes"]:
                logger.info("Choice: Yes")
                logger.info("Continuing to the next question!")
                logger.info("-------------------------------------")

                print("Choice: Yes")
                print("Continuing to the next question!")
                print("-------------------------------------")
                choice = True
    except Exception as error:
            logger.error("Error while running the chatbot: {}".format(error))
            print("Error while running the chatbot: {}".format(error))
            sys.exit()


if __name__ == "__main__":
    main()
