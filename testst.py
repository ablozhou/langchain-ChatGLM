import streamlit as st  #导入streamlit
import numpy as np

#form1 = st.form(key='my_form1')#key是form的关键字，不同form的key不能相同
#name = form1.text_input(label='Enter some text')
name = st.chat_input(placeholder="Your message",
                     key=None,
                     max_chars=None,
                     disabled=False,
                     on_submit=None,
                     args=None,
                     kwargs=None)
#submit_button = form1.form_submit_button(label='Submit',type='primary')

#
if name:
    st.write(f'hello {name}')

import pandas as pd

st.write("Here's our first attempt at using data to create a table:")
st.write(
    pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }))
dataframe = pd.DataFrame(np.random.randn(10, 20),
                         columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox('How would you like to be contacted?',
                                     ('Email', 'Home phone', 'Mobile phone'))

# Add a slider to the sidebar:
add_slider = st.sidebar.slider('Select a range of values', 0.0, 100.0,
                               (25.0, 75.0))
left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio('Sorting hat',
                      ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")