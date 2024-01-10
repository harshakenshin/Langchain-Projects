import streamlit as st
from QuestionAnswering import query_llm, initialise
import os
def main(file_name,ext="pdf"):
    st.title("document conversation agent")

    path = f"./docs/{file_name}.{ext}"
    obj = initialise(file_name,ext)
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
    
def save_uploaded_file(uploaded_file,ext="pdf"):
    read_name = uploaded_file.name 
    read_name = read_name.split(f".{ext}")[0]
    file_name = "-".join(read_name.split())
    path = f"./docs/{file_name}.{ext}"
    if not os.path.exists("./docs") :
        os.makedirs("./docs")
    if not os.path.exists('./db') :
        os.makedirs("./db")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return file_name


if __name__ == "__main__":
    uploaded_file_pdf = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
    uploaded_file_doc = st.sidebar.file_uploader("Upload a txt file", type="txt")
    if uploaded_file_pdf or uploaded_file_doc:
        if uploaded_file_pdf :
            file_name = save_uploaded_file(uploaded_file_pdf,ext="pdf")
            ext = "pdf"
        else :
            file_name = save_uploaded_file(uploaded_file_doc,ext="txt")
            ext = "txt"
        main(file_name,ext)
