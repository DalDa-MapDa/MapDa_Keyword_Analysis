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
OUTPUT_DIR = "result"
OUTPUT_EXCEL_PATH = os.path.join(OUTPUT_DIR, "emotion_analysis_result.xlsx")

# ───────── 프롬프트 정의 ─────────
# 문장이 긍정적이면 Positive, 부정적이면 Negative, 중립이면 Neutral로 응답
SYSTEM_PROMPT = (
    "다음 문장의 전체 맥락을 고려하여 감정 분석을 진행하세요. "
    "문장이 긍정적이면 Positive로, 부정적이면 Negative로, 중립이면 Neutral로 응답하세요. "
    "문장:"
)

# ───────── 감정 분석 API 응답 모델 정의 ─────────
class EmotionResultModel(BaseModel):
    result: str

# ───────── 감정 분석 함수 ─────────
def analyze_emotion(text: str) -> str:
    """
    Gemini API를 이용하여 주어진 텍스트의 감정 분석(긍정/부정/중립)을 수행합니다.
    
    Args:
        text (str): 분석할 문장
        
    Returns:
        str: "Positive", "Negative", "Neutral" 중 하나 (API 응답에 따라 결과가 없으면 None)
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

def main():
    # 결과 저장 폴더 생성 (없으면)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 원본 데이터 불러오기
    try:
        df_input = pd.read_excel(INPUT_EXCEL_PATH)
    except Exception as e:
        print(f"엑셀 파일({INPUT_EXCEL_PATH}) 읽기 오류: {e}")
        return
    
    # body 컬럼이 존재하는지 확인
    if "body" not in df_input.columns:
        print("엑셀 파일에 'body' 컬럼이 없습니다.")
        return

    # 2) 이미 처리된 결과가 있는지 확인
    if os.path.exists(OUTPUT_EXCEL_PATH):
        # 기존 결과 불러오기
        try:
            df_existing = pd.read_excel(OUTPUT_EXCEL_PATH)
        except Exception as e:
            print(f"기존 결과 파일({OUTPUT_EXCEL_PATH}) 읽기 오류: {e}")
            # 읽기 실패 시, 빈 데이터프레임으로 초기화
            df_existing = pd.DataFrame(columns=["no", "title", "body", "vote", "comment", "emotion_result"])
    else:
        # 결과 파일이 없으면 새로 생성
        df_existing = pd.DataFrame(columns=["no", "title", "body", "vote", "comment", "emotion_result"])
    
    # 3) 이미 처리된 no 목록 확인
    processed_nos = set(df_existing["no"].dropna().astype(int).tolist())

    # 4) 아직 처리되지 않은 행만 처리
    for idx, row in tqdm(df_input.iterrows(), total=len(df_input), desc="감정 분석 진행중"):
        # 현재 행의 no
        row_no = int(row["no"]) if not pd.isna(row["no"]) else None
        
        # 이미 처리된 no이면 스킵
        if row_no in processed_nos:
            continue
        
        # body가 NaN이면 None 처리
        body_text = row["body"] if not pd.isna(row["body"]) else None
        if not body_text:
            emotion_result = None
        else:
            # 감정 분석 수행
            emotion_result = analyze_emotion(str(body_text))
        
        # 결과를 df_existing에 추가
        new_row = {
            "no": row["no"],
            "title": row["title"],
            "body": row["body"],
            "vote": row["vote"],
            "comment": row["comment"],
            "emotion_result": emotion_result
        }
        # DataFrame으로 변환 후 concat
        df_new = pd.DataFrame([new_row])
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)

        # 혹시 중간에 순서가 뒤섞이지 않도록 원하는 컬럼 순서로 정렬
        required_columns = ["no", "title", "body", "vote", "comment", "emotion_result"]
        df_existing = df_existing[required_columns]

        # 중간 저장 (바로바로 저장)
        try:
            df_existing.to_excel(OUTPUT_EXCEL_PATH, index=False)
        except Exception as e:
            tqdm.write(f"중간 저장 오류: {e}")
            # 저장 실패 시에는 계속 진행하거나, 원하는 로직대로 처리 (여기서는 그냥 계속 진행)
    
    print(f"최종 감정 분석 결과가 '{OUTPUT_EXCEL_PATH}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
