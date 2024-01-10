import streamlit as st
from QuestionAnswering import query_llm, initialise
import os
def main(pdf_name):
    st.title("pdf conversation agent")

    # pdf_name = "salary"
    path = f"./docs/{pdf_name}.pdf"
    obj = initialise(pdf_name)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking..."):  # Show a thinking spinner while processing
            assistant_response = query_llm(prompt, obj)

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
    
def save_uploaded_file(uploaded_file):
    read_name = uploaded_file.name 
    read_name = read_name.split(".pdf")[0]
    pdf_name = "-".join(read_name.split())
    path = f"./docs/{pdf_name}.pdf"
    if not os.path.exists("./docs") :
        os.makedirs("./docs")
    if not os.path.exists('./db') :
        os.makedirs("./db")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return pdf_name


if __name__ == "__main__":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file :
        pdf_name = save_uploaded_file(uploaded_file)
        main(pdf_name)
