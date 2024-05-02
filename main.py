from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
import textwrap
import streamlit as st 
from streamlit_js_eval import streamlit_js_eval



# We basically first load the video for the given URL and then create a transcript for that video. But since the 
# transcript would have a large number of tokens, we cant send them to llm api directly. So instead we break them down into
# chunks of documents using the splitter. Then we use a similarity matcher in this Meta's FAISS which converts these doc chunks 
# into vectors of embeddings. Now when the user searches a query, it would find the most similar or relevant chunk from the chunks of
# these documents and we send that particular chunk information to the llm api

def create_db_from_youtube_video(video_url,embeddings):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs_chunks = text_splitter.split_documents(transcript)
    db_of_chunks = FAISS.from_documents(docs_chunks,embeddings)

    return db_of_chunks    

def get_response_for_query(db, query, openai_api_key, k=1):
    # do a similarity search for query and pick out the k best chunks
    docs = db.similarity_search(query, k=k)
    # convert the text in each chunk to one big text
    docs_content = " ".join([d.page_content for d in docs])

    # in chat model we have two prompts: system - to tell what the system needs to do and 
    # human - for conversing with the user/human
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

    system_template = """
        I would need your help today to answer questions about the music lyrics based on the youtube video's transcript : 
        {docs}
        Please try to make your answers succinct and to the point with maximum of 20 words.
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """Answer the following question: {question}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat,prompt=chat_prompt)
    response = chain.run(question = query, docs=docs_content)
    response = response.replace("\n","")
    return response, docs

def run():
        
    st.set_page_config(
        page_title="Youtube Assistant",
        page_icon="📽️",
    )

    st.title("Youtube Video Assistant")

    with st.sidebar:
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    st.write("### Please enter the link")
    video_link = st.text_input(label="video link", label_visibility="hidden")

    if st.session_state.openai_api_key and video_link:

        if 'db' not in st.session_state:
            st.session_state.db = None
        
        if st.session_state.db is None:
            st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
            st.session_state.db = create_db_from_youtube_video(video_link,st.session_state.embeddings)

        with st.form("my_form"):
            query = st.text_area("Enter your question:", "What is this song about?")
            submitted = st.form_submit_button("Submit")
            if submitted:
                response, docs = get_response_for_query(st.session_state.db,query=query,openai_api_key=st.session_state.openai_api_key,k=2)
                st.info(response)

    else:
        st.info("Please add your OpenAI API key and Youtube Video Link to continue.")

    if st.button('Reset and Reload App'):
        # for key in list(st.session_state.keys()):
        #     del st.session_state[key]
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        # st.rerun()


if __name__ == "__main__":
    run()
