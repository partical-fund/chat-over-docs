import os
from typing import Tuple, List
from operator import itemgetter
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import utils

os.environ['OPENAI_API_KEY'] = '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'

def get_answer(path, user_input):
	loader = DirectoryLoader(path, glob="**/*.pdf")

	docs = loader.load()

	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

	texts = text_splitter.split_documents(docs)

	texts = langchain.vectorstores.utils.filter_complex_metadata(texts)

	embeddings = OpenAIEmbeddings()

	vectorstore = Chroma.from_documents(texts, embeddings)

	retriever = vectorstore.as_retriever()

	llm = OpenAI(temperature=0)

	

	return result

while True:
	path_name = input("Enter path name: ")
	print("Ask a question...")
	while True:
		user_input = input("You: ")
		response = get_answer(path_name, user_input)
		print("Response: ", response)
