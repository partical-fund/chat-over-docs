import os
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ['OPENAI_API_KEY'] = <INSERT KEY>

def make_retriever(path_name):
	
	loader = DirectoryLoader(path_name, glob="**/*.pdf")

	docs = loader.load()

	text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=256, chunk_overlap=0
	)
	
	texts = text_splitter.split_documents(docs)

	texts = langchain.vectorstores.utils.filter_complex_metadata(texts)

	embeddings = OpenAIEmbeddings()

	vectorstore = Chroma.from_documents(texts, embeddings)

	retriever = vectorstore.as_retriever()

	return retriever

def get_answer(user_input, chat_history, retriever):

	llm = ChatOpenAI(temperature=0, model_name="gpt-4")

	_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

	Chat History:
	{chat_history}
	Follow Up Input: {question}
	Standalone question:"""
	
	CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

	question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

	doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
	
	chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain
	)

	result = chain({"question": user_input, "chat_history": chat_history})

	return result["answer"]

while True:
	path_name = input("Enter path name: ")
	retriever = make_retriever(path_name)
	print("Ask a question...")
	chat_history = []
	while True:
		user_input = input("You: ")
		response = get_answer(user_input, chat_history, retriever)
		chat_history = [(user_input, response)]
		print("Response: ", response)
