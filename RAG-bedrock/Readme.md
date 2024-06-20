

## RAG 실습

#### 실습 내용

아래 jupyter notebook을 순서대로 실행 해 주세요.

- `01-bedrock-test.ipynb` : bedrock의 기본적인 사용 방법을 실습 해 봅니다.
- `02-aoss-creation.ipynb` : Opensearch의 기본적인 사용 방법을 실습 해 봅니다.
- `03-data-prep-indexing.ipynb` : 데이터를 준비하고 인덱싱하는 기본적인 방법을 실습 해 봅니다.
- `04-rag-simple.ipynb` : 가장 기본적인 형태의 RAG 원리를 실습 해 봅니다.


#### Bedrock 활용 

bedrock을 활용하는 것은 주로 아래의 2가지 방법이 사용됩니다.
- `boto3` : 기본적인 AWS Python SDK 이기 때문에 다양한 customization이 가능합니다.
- `langchain` : [langchain](https://www.langchain.com/) 은 LLM app 개발 시 많이 활용되는 라이브러리입니다. prototype 이나 demo를 빠르게 만들고 싶은 경우 적합합니다. 다만 customization 을 하려면 직접 코드를 수정해야 하는 경우가 많습니다.
- langchain을 사용하게 되면 abstraction이 많이 되어 있어서 내부 동작을 자세히 알기 어려울 수 있습니다. 따라서 이 실습에서는 langchain 없이 기본적이 ㄴRAG가 어떻게 동작하는지 살펴봅니다.



## 참고자료

- Advanced RAG 샘플코드 : [sample code](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot)
- RAG application 예시 : [streamlit code](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/04_web_ui)
- 복잡한 문서에 RAG 적용 : [sample code](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/10_hands_on_lab/02_rag_over_complex_pdf/01_rag_over_complex_doc.ipynb)
- Serverless RAG app 개발 : [code](https://github.com/sungeuns/ultimate-rag), [배포 가이드](https://sungeuns.github.io/ultimate-rag/guide/deploy.html), [개발 가이드](https://sungeuns.github.io/ultimate-rag/guide/guide-ko.html)