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

	memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer')
	
	_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

	Chat History:
	{chat_history}
	Follow Up Input: {question}
	Standalone question:"""

	CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

	template = """Answer the question based only on the following context:
	{context}

	Question: {question}
	
	"""
	ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

	DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

	def _combine_documents(docs, document_prompt = DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
		doc_strings = [format_document(doc, document_prompt) for doc in docs]
		return document_separator.join(doc_strings)

	def _format_chat_history(chat_history: List[Tuple]) -> str:
		buffer = ""
		for dialogue_turn in chat_history:
			human = "Human: " + dialogue_turn[0]
			ai = "Assistant: " + dialogue_turn[1]
			buffer += "\n" + "\n".join([human, ai])
		return buffer

	_inputs = RunnableMap(
    	standalone_question=RunnablePassthrough.assign(
        	chat_history=lambda x: _format_chat_history(x['chat_history'])
		) | CONDENSE_QUESTION_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser(),
	)
	_context = {
    	"context": itemgetter("standalone_question") | retriever | _combine_documents,
    	"question": lambda x: x["standalone_question"]
	}

	loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
	)

	standalone_question = {
    	"standalone_question": {
        	"question": lambda x: x["question"],
        	"chat_history": lambda x: _format_chat_history(x['chat_history'])
    	} | CONDENSE_QUESTION_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser(),
	}

	retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"]
	}
	
	final_inputs = {
		"context": lambda x: _combine_documents(x["docs"]),
		"question": itemgetter("question")
	}
	
	answer = {
		"answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
		"docs": itemgetter("docs"),
	}
	
	final_chain = loaded_memory | standalone_question | retrieved_documents | answer
	
	inputs = {"question": user_input}
	result = final_chain.invoke(inputs)
	result

	return result

while True:
	path_name = input("Enter path name: ")
	print("Ask a question...")
	while True:
		user_input = input("You: ")
		response = get_answer(path_name, user_input)
		print("Response: ", response)
