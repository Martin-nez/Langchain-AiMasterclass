from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model1 = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

model2 = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0.7
)

prompt_1 = PromptTemplate.from_template("Generate short and simple note for the following text {text}")

prompt_2 = PromptTemplate.from_template("Generate 5 short question answer from the following text {text} ")

prompt_3 = PromptTemplate.from_template("Merge the provided notes and question answer pairs into a single comprehensive document. {notes} {qa_pairs}")

parser = StrOutputParser()

chain_1 = prompt_1 | model1 | parser

chain_2 = RunnableSequence(prompt_2, model2, parser) # We can use this too in the normal chaining operator | to chain multiple runnables together in a sequence. This allows us to more complex chains that can handle multiple steps of processing.

runnable_chain = RunnableParallel({
    "notes" : chain_1,
    "qa_pairs" : chain_2
})

merge_gain = prompt_3 | model1 | parser

final_chain = RunnableSequence(runnable_chain, merge_gain)

text = """Artificial intelligence is rapidly transforming the way we live and work. From virtual assistants and recommendation systems to advanced medical diagnostic, AI is becoming deeply integrated into everyday life. Businesses use machine learnng models to analyze data, improve customer experiences, and automate repetitive tasks. In education, AI-powered tools personalize learning and provide instant feedback. However, with this advancements come important questions about ethics, privacy, and job displacement. Responsible development and transparent policies are essential to ensure technology benefits everyone. As innovation continues to accelerate, understanding how AI works and how it impacts society is more important than ever before."""

response = final_chain.invoke({"text": text})

print(response)