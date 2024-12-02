from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from Intelligence.ai_engine import groq_api

# Initializing model for inference.
model = groq_api()

# Human template.

human_template = """
        You are an AI that performs sentiment analysis. Analyze the following text and provide the sentiment:
        Text: "{question}"
        Output formate:
        (sentiment)
        Instructions:
        - Do not include any text other than the sentiment.
        - Return only the sentiment in the output.
        - Do not hallucinate.
        - Think before decision making, and always provide correct answer.
        - Do not make variations in sentiment, make your answer final.
        - Only add 'Positive', 'Negative','Irrelevant' and 'Neutral' word.
        """


# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", human_template)
    ]
)


# chain.
chain = prompt_template | model

# Session store. session_id : {chat_history}
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def chat_with_history(process_chain=chain):
    return RunnableWithMessageHistory(
        process_chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )


# chat = chat_with_history()

# while True:
#     user_question = input(">>>>>")
#     result = chat.invoke(
#         {"question": user_question},
#         config={"configurable": {"session_id": "foo"}},
#
#     )
#
#     print(result.content)
