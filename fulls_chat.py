import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
# 全滚动条
st.set_page_config(page_title="智能客服")

# Sidebar contents
# with st.sidebar:
#     st.title('智能客服')
#     st.markdown('''
#     ## About
#     This app is an LLM-powered chatbot built using:
#     - [Streamlit](https://streamlit.io/)
#     - [HugChat](https://github.com/Soulter/hugging-chat-api)
#     - [OpenAssistant/oasst-sft-6-llama-30b-xor](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) LLM model

#     💡 Note: No API key required!
#     ''')
#     add_vertical_space(5)
#     st.write(
#         'Made with ❤️ by [Data Professor](https://youtube.com/dataprofessor)')

# Generate empty lists for chatbot and user.
## chatbot stores AI chatbot responses
if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = ["您好"]
## user stores User's questions
if 'user' not in st.session_state:
    st.session_state['user'] = ["你好"]

# Layout of input/response containers

response_container = st.container()
#colored_header(label='', description='', color_name='blue-30')
input_container = st.container()


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.chat_input(placeholder="开始聊天", key="input")
    return input_text


## Applying the user input box
with input_container:
    user_input = get_text()


# Response output
## Function for taking user prompt as input followed by producing AI chatbot responses
def generate_response(prompt):
    #chatbot = hugchat.ChatBot()
    #chatbot.chat(prompt)
    response = f"{prompt}"
    return response


## Conditional display of AI chatbot responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.user.append(user_input)
        st.session_state.chatbot.append(response)

    if st.session_state['chatbot']:
        for i in range(len(st.session_state['chatbot'])):
            message(st.session_state['user'][i],
                    is_user=True,
                    key=str(i) + '_user')
            message(st.session_state["chatbot"][i], key=str(i))