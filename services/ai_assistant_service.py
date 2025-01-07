import os 
import openai
import streamlit as st
import time as t
from services.ai_prompts_service import AIPrompts,PromptType,InstructionType
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import HybridFusion


openai.api_key = os.environ["OPENAI_API_KEY"]


wcd_url = os.environ["WEAVIATE_URL"]
wcd_api_key = os.environ["WEAVIATE_KEY"]


class AssistantService:
    def __init__(self, chat_history: list) -> None:
        self.chat_history = chat_history
        self.llm = openai.OpenAI()
        self.vectorstore_client = self.get_vectorstore_client()
        self.session_history = chat_history
        self.session_history_summary = None
        self.adjustment_data = None

    def get_session_history(self) -> list:
        return self.session_history
    
    def contextualize_user_query(self,query: str):
        contextualize_q_system_prompt = "Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        
        combined_history = [self.session_history_summary] + self.session_history

        messages = [
            {"role":"system", "content":contextualize_q_system_prompt},
            {"role":"user","content": 
            f"""
            User Question: {query}
            Chat history: {combined_history}
            Formulated Standalone Question:
             """},
        ]

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        return response.choices[0].message.content
    
    def get_vectorstore_client(self):
        connection = weaviate.connect_to_weaviate_cloud(
            cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
            auth_credentials=Auth.api_key(wcd_api_key),  # Replace with your Weaviate Cloud key
            headers={'X-OpenAI-Api-key': os.environ["OPENAI_API_KEY"]})
        
        client = connection.collections.get("Content")

        return client
    
    def get_relevant_documents(self,query):
        relevant_documents = self.vectorstore_client.query.hybrid(
            query=query,
            limit=12,
            auto_limit=1,
            fusion_type=HybridFusion.RELATIVE_SCORE)

        documents = [doc.properties['content'] for doc in relevant_documents.objects]

        return documents
    
    def summarize_session_history(self) -> str:

        system_prompt = AIPrompts.get_prompt(PromptType.SESSION_HISTORY_SUMMARIZATION_PROMPT,InstructionType.ASSISTANT,{})
        
        history_messages = self.session_history[-1] if self.session_history_summary else self.session_history

        user_prompt = AIPrompts.get_prompt(PromptType.SESSION_HISTORY_SUMMARIZATION_PROMPT,InstructionType.USER,{"messages":history_messages})

        messages = [
            {"role":"system", "content":system_prompt},
            {"role":"user","content": user_prompt},
        ]

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        if self.session_history_summary:
            self.session_history_summary = self.session_history_summary + '\n' + response.choices[0].message.content
        else:
            self.session_history_summary = response.choices[0].message.content
            
    def update_session_history(self) -> None:
        if len(self.chat_history) > 1:
            self.summarize_session_history()
            st.session_state.session_history_summary = self.session_history_summary
            self.session_history = self.session_history[-2:]
    
    def update_history(self, question: str, answer: str) -> None:
        self.chat_history.append((f"User: {question}", f"Assistant: {answer}"))
        self.session_history.append((f"User: {question}", f"Assistant: {answer}"))

        self.update_session_history()

    def decode_image(self, image_64b: str, file_format: str):
        messages = [
            {"role":"system", "content":
             """
            # Instructions
            You are an expert at decoding manufacturing process and system questions. Analyse the contents of an image extensively and determine what is being asked of the user, focusing on manufacturing processes. Your analysis should include identifying the subject, outlining the elements present, highlighting key parts and features, and noting any relevant points for further analysis. Of course, include the key question being asked.
             
            # Steps
            - **Image Identification**: Specify what the image depicts, including the setting and any objects or machinery.
            - **Element Description**: Describe any diagrams, text, or labels present in the image, highlighting their significance to the manufacturing process.
            - **Feature Highlighting**: Identify and detail key parts and features evident in the image. Explain their role and importance in the process.
            - **Question Extraction**: Extract the question word-for-word. If the question is a mutliple choice question, include the provided options and the numbers/letters next to each option
            - **Design Analysis**: Offer insights into the design aspects of the image, focusing on functionality and efficiency.
            - **Relevant Points**: Note any critical observations that could aid in analyzing or improving the process.

            # Response Format
            Adhere the the below response format:

            Questions: [describe the main questions extracted here. If it is a mutliple choice question, include the provided options and the numbers/letters next to each option]
            Details: The entire analysis


             """},
            {"role":"user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{file_format};base64,{image_64b}"},
                    }
                ],}
        ]

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        return response.choices[0].message.content

        
    
    def send_input(self,user_input: str, box, prompt_type: PromptType,image_sent: bool = False) -> str:

        contextualized_input = self.contextualize_user_query(user_input)

        relevant_content = self.get_relevant_documents(contextualized_input)

        combined_history = [st.session_state.session_history_summary] + self.session_history

        combined_dict_data = {"chat_history":combined_history,"book_content":relevant_content}

        system_prompt = AIPrompts.get_prompt(prompt_type,InstructionType.ASSISTANT,combined_dict_data)
        
        # user_prompt = AIPrompts.get_prompt(prompt_type,InstructionType.USER,{"chat_history":combined_history,"user_input":user_input})

        messages = [
            {"role":"system", "content":system_prompt},
            {"role":"user","content": f"""
             
             # Question
             <question>
             {contextualized_input}
             </question>

             Take in all relevant details, but answer the questions only, and provide extensive reasoning and in-depth rationale for your answers. If the questions are multiple choice, then answer based on the choices given.
             """},
        ]

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True
        )

        collective_response = ""
        for chunk in response:
            if chunk.choices[0].finish_reason == "stop":
                break
            for c in chunk.choices[0].delta.content:
                collective_response += c
                t.sleep(.002)
                box.write(collective_response)

        self.update_history(user_input,collective_response)        

        return collective_response
    