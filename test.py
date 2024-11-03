# chains/ext_and_gen_test.py 파일. 키워드 추출 후 답변 및 문제 생성하는 test 파일.

# langchain_env 폴더를 sys.path에 추가
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# KeyBert를 이용한 keyword 추출 클래스
# from models.bert.keyBert_test import Ext_Key
from models.BERT.keyBert_main import Ext_Key

# from models.llm.bllossom_test import LLMModel_BLLOSSOM
from models.LLM.gpt import LLMModel_GPT

# 자체 구축한 KeywordChain 
class KeywordChain:
    def __init__(self, filepath):
        self.filepath = filepath

        self.Key_Obj = Ext_Key(self.filepath) # 변경된 부분 ->  KeyBERT 인스턴스를 생성하고 밑에서 함수를 호출해야 함. 
        # self.BLl_Obj = LLMModel_BLLOSSOM()  
        self.GPt_Obj = LLMModel_GPT()

    # BLLossom LLM을 사용하는 함수
    # def ext_and_gen_BLl(self):
    #     """텍스트에서 키워드를 추출하고 Bllossom 프롬프트로 전달하여 문장 생성"""
    #     # 1. 키워드 추출 후 확인
    #     keywords = self.Key_Obj.keyword_ext()
    #     print("추출된 키워드:", keywords) 
        
    #     # 2. 키워드 추출 후 GNN에 입력 -> 아직 안됨. 


    #     # 3. 추출 후 프롬프트로 전달 -> 현재는 GNN (x)
    #     response = self.BLl_Obj.generate_response(keywords)
    #     print("추출된 해설 및 문제 : ")
        
    #     # 응답 생성
    #     return response
    
    # GPT API 를 사용하는 함수
    def ext_and_gen_GPt(self):
        """텍스트에서 키워드를 추출하고 GPT 프롬프트로 전달하여 문장 생성"""

        # 1. 키워드 추출
        keywords = self.Key_Obj.keyword_ext()
        print("1. 추출된 키워드:", keywords)

        # 2. 키워드 추출 후 GNN에 입력 -> 관계 도출이 안됨.


        # 3. 추출된 관계도와 용어를 LLM에 전달. -> 현재는 GNN (x)
        response = self.GPt_Obj.generate_response(keywords)
        print("2. 추출된 해설 및 문제: ")

        # 응답 생성
        return response


# 응답 test 함수
def test():
    # data/articles 의 sample.html 파일 
    filepath = 'data\\articles\\sample2.html'

    # 모델 인스턴스 생성
    example = KeywordChain(filepath)

    # GPT API를 통한 응답 생성
    response = example.ext_and_gen_GPt()
    print(response)

# if __name__ == "__main__":
#     test()