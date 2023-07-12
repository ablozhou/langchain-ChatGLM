import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html

st.title("智能客服")
a = '''
您这会可能在忙呢，如果有需要帮助或者咨询的，可以随时联系我们哈。感谢您的支持，祝您生活愉快~~
<h1>智能客服</h1>

AvatarStyle = Literal[
    "adventurer",
    "adventurer-neutral",
    "avataaars",
    "big-ears",
    "big-ears-neutral",
    "big-smile",
    "bottts",
    "croodles",
    "croodles-neutral",
    "female",
    "gridy",
    "human",
    "identicon",
    "initials",
    "jdenticon",
    "male",
    "micah",
    "miniavs",
    "pixel-art",
    "pixel-art-neutral",
    "personas",
    https://api.dicebear.com/5.x/icons/svg?seed=44&rotate=180
    winsky
]
'''


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    res = {'type': 'normal', 'data': user_input}
    st.session_state.generated.append(res)


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


st.session_state.setdefault(
    'past',
    [
         'plan text with line break'
    ])
st.session_state.setdefault(
    'generated',
    [
        {
            'type': 'normal',
            'data': '您好'
        },
    ])

chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])):
        
        message(st.session_state['past'][i],
                is_user=True,
                avatar_style='icons',
                seed='44&rotate=180',
                key=f"{i}_user")
        message(
            st.session_state['generated'][i]['data'],
            key=f"{i}",
            avatar_style='icons',  #icons
            seed='43',
            allow_html=True,
            is_table=True
            if st.session_state['generated'][i]['type'] == 'table' else False)

    #st.button("Clear message", on_click=on_btn_click)

with st.container():
    input_text = st.chat_input(placeholder="开始聊天",
                               on_submit=on_input_change,
                               key="user_input")
    # st.text_input("User Input:", on_change=on_input_change, key="user_input")