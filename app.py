import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.callbacks import get_openai_callback


st.title("🦜🔗 Youtuber")
prompt = st.text_input("Enter the topic:")
use_wiki = st.checkbox('Use Wikipedia Research')

#prompt template
title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a youtube video title about {topic}."
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template="Write me a youtube video script based on title TITLE: {title}. Also, do some research on {wikipedia_research} and include it in the script."
)

#memory
title_memory = ConversationBufferMemory(input_key='topic',
                                  memory_key='chat_memory')
script_memory = ConversationBufferMemory(input_key='title',
                                  memory_key='chat_memory')


#llms
llm = OpenAI(model='text-babbage-001', temperature=0.9)
title_chain = LLMChain(llm=llm,
                       prompt=title_template,
                       output_key='title',
                       verbose=True,
                       memory=title_memory)
script_chain = LLMChain(llm=llm,
                        prompt=script_template,
                        output_key='script',
                        verbose=True,
                        memory=script_memory)
# sequential_chain = SequentialChain(chains=[title_chain, script_chain],
#                                    input_variables=['topic'],
#                                    output_variables=['title', 'script'],
#                                    verbose=True)

wiki = WikipediaAPIWrapper()

#if there is a prompt, run call llm
if prompt:
    with get_openai_callback() as cb:
        title = title_chain.run(topic=prompt)
        if use_wiki:
            wiki_research = wiki.run(prompt)
        else:
            wiki_research = ''
        script = script_chain.run(title=title, wikipedia_research=wiki_research)
#         response = sequential_chain({'topic': prompt})
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
    
    st.write(title)
    st.write(script)
    
#     st.write(response['title'])
#     st.write(response['script'])

    with st.expander("Title History"):
        st.write(title_memory.buffer)

    with st.expander("Script History"):
        st.write(script_memory.buffer)
    
    with st.expander("Wikipedia Research"):
        st.write(wiki_research)
