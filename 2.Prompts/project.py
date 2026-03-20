from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.9)

print("Blog Post Generator")
print("Provide ideas or topics for the blog post, Type exit to finish.\n")

user_topic = input("Enter Blog post topic:")


# Create initial system prompt
system_prompt = ChatPromptTemplate.from_messages([
("system", "You are a professional blog post writer. Help generate informative, engaging and well-structured blog post about {topic}."),
("human", "write a detailed blog post about {topic}.")
])

chat_history = []

while True:
    user_input = input("Ideas or instruction or type exit to finish:")
    if user_input.lower() =="exit":
        print("Exiting the blog post")
        break

    
    #build mesages: system prompt + history + current input
    messages = system_prompt.format_messages(topic = user_topic)
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_input))

    response = chat_model.invoke(messages)
    print("\nBlog Post Content:\n")
    print(response.content)
    print("\n", + "="*50, +  "\n")


    #update chat history
    chat_history.append(HumanMessage(content= user_input))
    chat_history.append(AIMessage(content=response.content))
    for message in chat_history:
        print(f"{message.type}: {message.content}")

