import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

os.environ['OPENAI_API_KEY'] = apikey

st.title("🦜🔗 Expermiment")
prompt = st.text_input("Enter your prompt:")

#prompt template
title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a youtube video title about {topic}."
)

script_template = PromptTemplate(
    input_variables=['title'],
    template="Write me a youtube video script based on title TITLE: {title}."
)

#memory
memory = ConversationBufferMemory(input_key='topic',
                                  memory_key='chat_memory')


#llms
llm = OpenAI(model='text-babbage-001', temperature=0.9)
title_chain = LLMChain(llm=llm,
                       prompt=title_template,
                       output_key='title',
                       verbose=True,
                       memory=memory)
script_chain = LLMChain(llm=llm,
                        prompt=script_template,
                        output_key='script',
                        verbose=True,
                        memory=memory)
sequential_chain = SequentialChain(chains=[title_chain, script_chain],
                                   input_variables=['topic'],
                                   output_variables=['title', 'script'],
                                   verbose=True)

#if there is a prompt, run call llm
if prompt:
    with get_openai_callback() as cb:
        response = sequential_chain({'topic': prompt})
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
    
    st.write(response['title'])
    st.write(response['script'])

    with st.expander("Memory History"):
        st.write(memory.buffer)
