import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

# ───────── 환경변수 및 API 키 설정 ─────────
load_dotenv()  # 루트 디렉토리의 .env 파일 로드
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# ───────── 파일 경로 변수 설정 ─────────
INPUT_EXCEL_PATH = os.path.join("data", "school.xlsx")
OUTPUT_EXCEL_PATH = os.path.join("result", "emotion_analysis_result.xlsx")

# ───────── 프롬프트 정의 ─────────
SYSTEM_PROMPT = (
    "다음 문장의 전체 맥락을 고려하여 감정 분석을 진행하세요. "
    "문장이 긍정적이면 Positive로, 부정적이면 Negative로, 중립이면 Neutral로 응답하세요. "
    "문장:"
)

# ───────── 감정 분석 API 응답 모델 정의 ─────────
class EmotionResultModel(BaseModel):
    result: str

# ───────── 감정 분석 함수 ─────────
def analyze_emotion(text):
    """
    Gemini API를 이용하여 주어진 텍스트의 감정 분석(긍정/부정/중립)을 수행합니다.
    
    Args:
        text (str): 분석할 문장
        
    Returns:
        str: "Positive", "Negative", 또는 "Neutral" (API 응답에 따라 결과가 없으면 None)
    """
    try:
        prompt = f"{SYSTEM_PROMPT} {text}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config={
                'response_mime_type': 'application/json',
                'response_schema': EmotionResultModel,
            }
        )
        # Gemini API의 파싱된 결과가 존재하는 경우 사용
        if response.parsed is not None:
            result_obj = response.parsed  # EmotionResultModel 인스턴스
            result = result_obj.result
        else:
            # 파싱 실패 시, API 응답 문자열로부터 수동 파싱 시도
            result_text = response.text.strip()
            try:
                result_json = json.loads(result_text)
                result = result_json.get("result", None)
            except Exception:
                # JSON 파싱 실패 시, "Positive"/"Negative"/"Neutral" 단어 포함 여부로 판별
                if "Positive" in result_text:
                    result = "Positive"
                elif "Negative" in result_text:
                    result = "Negative"
                elif "Neutral" in result_text:
                    result = "Neutral"
                else:
                    result = result_text
        return result
    except Exception as e:
        tqdm.write(f"Gemini API 호출 중 오류 발생: {e}")
        return None

# ───────── 메인 처리 로직 ─────────
def main():
    # Excel 파일 읽기
    try:
        df = pd.read_excel(INPUT_EXCEL_PATH)
    except Exception as e:
        print(f"엑셀 파일 읽기 오류: {e}")
        return
    
    # body 컬럼이 존재하는지 확인
    if "body" not in df.columns:
        print("엑셀 파일에 'body' 컬럼이 없습니다.")
        return

    # 각 body 텍스트에 대해 감정 분석 진행
    emotion_results = []
    for text in tqdm(df["body"].tolist(), desc="감정 분석 진행중"):
        if pd.isna(text):
            emotion_results.append(None)
        else:
            result = analyze_emotion(str(text))
            # 결과 문자열(Positive, Negative, Neutral 등)만 저장
            emotion_results.append(result if result else None)
    
    # 새로운 컬럼 emotion_result 추가
    df["emotion_result"] = emotion_results

    # 최종적으로 저장할 컬럼: no, title, body, vote, comment, emotion_result
    required_columns = ["no", "title", "body", "vote", "comment", "emotion_result"]
    df_to_save = df[required_columns]
    
    # 결과 저장 폴더 생성 (없으면)
    os.makedirs(os.path.dirname(OUTPUT_EXCEL_PATH), exist_ok=True)
    
    # Excel 파일로 저장
    try:
        df_to_save.to_excel(OUTPUT_EXCEL_PATH, index=False)
        print(f"감정 분석 결과가 {OUTPUT_EXCEL_PATH}에 저장되었습니다.")
    except Exception as e:
        print(f"엑셀 파일 저장 오류: {e}")

if __name__ == "__main__":
    main()
