from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import load_chat_planner, load_agent_executor, PlanAndExecute
from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import torch
from medical import lora_clip
from food_clip import predict_top_class
import os
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

api_key = "api_key"
# Load CLIP model and processor
def classify_image(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = lora_clip()
    images = preprocess(image).unsqueeze(0).to(device)
    cls = predict_top_class(model, images)
    return cls

# PDF 데이터 로드 및 벡터화
def load_pdf_vectorstore(pdf_file_path):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    loader = PyPDFLoader(pdf_file_path)  # 단일 PDF 파일 로드
    documents = loader.load_and_split(text_splitter)  # 텍스트 분할
    vectorstore = FAISS.from_documents(documents, embeddings)  # 벡터스토어 생성
    return vectorstore

# 에이전트 초기화
def load_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
    tools = load_tools(tool_names=["ddg-search"], llm=llm)

    # PDF 데이터 준비
    pdf_directory = "./Glycemic_Index.pdf"  # 제공한 PDF 파일 경로
    vectorstore = load_pdf_vectorstore(pdf_directory)

    # 메모리 설정
    memory = ConversationBufferMemory(memory_key="conversation", return_messages=True)

    # 프롬프트 설정
    planner = load_chat_planner(llm)
    executor = load_agent_executor(llm, tools, verbose=True)

    # 에이전트 실행 함수
    class PlanAndExecuteAgent:
        def __init__(self, planner, executor, memory, vectorstore):
            self.agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
            self.memory = memory
            self.vectorstore = vectorstore

        def run(self, inputs):
            try:
                # 이미지 분류
                image_class = None
                if inputs.get("image"):
                    image_class = classify_image(inputs["image"])
                    print(image_class)

                    query = image_class
                    pdf_results = self.vectorstore.similarity_search(query, k=3)
                    pdf_context = "\n".join([doc.page_content for doc in pdf_results])
                memory_variables = self.memory.load_memory_variables({})
                past_conversation = memory_variables.get("conversation", "")

                # 기존 프롬프트 유지
                prompt = """
                Translate the input question into English if necessary.  
                The response MUST strictly follow the specified output format.  
                The response must be concise, clear, and written EXCLUSIVELY in Korean.  
                Any deviation from the format will be considered incorrect.
                
                ---
                
                ### Mandatory Requirements:
                1. You must respond by following the given output example, including Nutritional Information, Glycemic Index, Advantages, Disadvantages, and Friendly Advice.  
                2. The response must be written exclusively in Korean.                  
                ---
                
                ### Output Format Example:
                **Nutritional Information**:
                    - Calories: [number] kcal
                    - Protein: [number] g
                **Glycemic Index (GI)**: [value]
                **Advantages**:
                    - [First advantage]
                    - [Second advantage]
                **Disadvantages**:
                    - [First disadvantage]
                    - [Second disadvantage]
                **Friendly Advice**:
                    - [Write the advice here]
            
                ---
                
                ### Cautions:
                - You must strictly follow the output format example above and respond accurately in Korean with all sections: Nutritional Information, Glycemic Index, Advantages, Disadvantages, and Friendly Advice.
                """
                translated_prompt = "The final output should be translated into Korean."
                memory_prompt = f"{inputs['question']} 중심으로 답변을 해줘"

                if not past_conversation==[]:
                # 현재 질문 구성
                    if image_class:
                        full_question = f"User: {inputs['question']}(Image class: {image_class})"
                        full_question = full_question + memory_prompt + prompt

                    else:
                        full_question = f"User: {past_conversation}\n{inputs['question']}"
                        full_question = full_question + memory_prompt + translated_prompt

                else:
                    if image_class:
                        full_question = f"User: {inputs['question']}(Image class: {image_class})"
                        full_question = full_question + prompt

                    else:
                        full_question = f"User: {inputs['question']}"
                        full_question = full_question + translated_prompt

                # 전체 입력 구성
                if image_class:
                    final_input = f"""
                                    User's Question: {full_question}
                                    PDF Search Results: {pdf_context}
                                    """
                    final_input += f"Image Classification Result: {image_class}"
                else:
                    final_input = f"""
                                    User's Question: {full_question}
                                    """

                # 에이전트 실행
                result = self.agent.run({"input": final_input})

                # 메모리 업데이트
                self.memory.save_context(
                    {"user_input": inputs["question"]+f"(Image class: {image_class})"},
                    {"response": result}
                )
                return result

            except Exception as e:
                print(f"Unexpected Error: {e}")
                return "오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    return PlanAndExecuteAgent(planner, executor, memory, vectorstore)