import gradio as gr
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def create_conversation() -> ConversationalRetrievalChain:

    embed_dir = 'embed_dir'

    embed = OpenAIEmbeddings(
        openai_api_key="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"
    )

    embed_dir = Chroma(
        persist_directory=embed_dir,
        embedding_function=embed
    )

    memory_hist = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )

    conv = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k',openai_api_key="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"),
        chain_type='stuff',
        retriever=embed_dir.as_retriever(),
        memory=memory_hist,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return conv

qa = create_conversation()


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def greet(name):
    return "Hello I am Saatva's AI assistant, here to assist you with any questions about our products. Feel free to ask me anything related to Saatva, and I will be happy to help. Happy shopping!!"

def bot(history):
    res = qa(
        {
            'question': history[-1][0],
            'chat_history': history[:-1]
        }
    )
    history[-1][1] = res['answer']
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot",
                         label='Document GPT').style(height=750)
    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=1):
            submit_btn = gr.Button(
                'Submit',
                variant='primary'
            )
        with gr.Column(scale=1):
            clear_btn = gr.Button(
                'Clear',
                variant='stop'
            )

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

    submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    demo.queue(concurrency_count=3)
    demo.launch(share=True)
