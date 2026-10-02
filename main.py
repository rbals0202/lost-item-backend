from typing import Annotated
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil


try:
    from ai_matcher import match_lost_item
except ModuleNotFoundError as e:
    if e.name == "ai_matcher":
        match_lost_item = None
    else:
        raise


app = FastAPI()


# 프론트엔드와 외부 서버 연동 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 로컬에서는 uploads
# 공개 서버에서는 환경변수로 다른 저장 위치 지정 가능
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 공개 서버 주소
# 예: https://lost-item-backend.onrender.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)


def make_image_url(request: Request, image_path: str):
    filename = os.path.basename(image_path)

    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/uploads/{quote(filename)}"

    return str(
        request.url_for(
            "uploads",
            path=filename
        )
    )


def get_test_candidates(image_path):
    return [
        {
            "item_id": "item_001",
            "category": "wallet",
            "description": "검은색 가죽 카드지갑",
            "images": [
                image_path,
                image_path
            ],
            "location": "학생회관",
            "datetime": "2026-09-21T11:00:00"
        }
    ]


@app.get("/")
def home():
    return {
        "message": "Lost Item Backend Server Running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/lost-items")
def create_lost_item(
    request: Request,
    image: Annotated[list[UploadFile], File()],
    description: str = Form(...),
    location: str = Form(...),
    datetime: str = Form(...)
):

    if len(image) < 1 or len(image) > 5:
        return {
            "status": "error",
            "message": "사진은 1장 이상 5장 이하로 업로드해야 합니다."
        }

    image_paths = []

    for uploaded_image in image:

        original_filename = os.path.basename(uploaded_image.filename)

        extension = os.path.splitext(original_filename)[1]

        saved_filename = f"{uuid4().hex}{extension}"

        image_path = os.path.join(
            UPLOAD_DIR,
            saved_filename
        )

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(
                uploaded_image.file,
                buffer
            )

        image_paths.append(image_path)


    # 현재 AI는 분실물 이미지 1장만 사용
    # 프론트에서 받은 1~5장은 모두 저장하고
    # AI에는 우선 첫 번째 사진을 전달
    lost_item = {
        "image_path": image_paths[0],
        "description": description,
        "location": location,
        "datetime": datetime
    }


    # 현재는 실제 DB 데이터 대신
    # AI 연동 확인용 후보 데이터 사용
    candidates = get_test_candidates(
        image_paths[0]
    )


    if match_lost_item is None:
        return {
            "status": "waiting_for_ai",
            "message": "ai_matcher.py 파일이 없습니다."
        }


    results = match_lost_item(
        lost_item,
        candidates
    )


    candidate_map = {
        candidate["item_id"]: candidate
        for candidate in candidates
    }


    frontend_results = []


    for result in results:

        candidate = candidate_map[
            result["item_id"]
        ]

        candidate_image_url = make_image_url(
            request,
            candidate["images"][0]
        )


        frontend_results.append({
            "item_id": result["item_id"],

            "description": candidate["description"],

            "image_url": candidate_image_url,

            "location": candidate["location"],

            "datetime": candidate["datetime"],


            "matching_score": round(
                max(
                    0.0,
                    min(
                        1.0,
                        result["matching_score"]
                    )
                ) * 100,
                1
            ),


            "image_similarity": round(
                max(
                    0.0,
                    min(
                        1.0,
                        result["image_similarity"]
                    )
                ) * 100,
                1
            ),


            "text_similarity": round(
                max(
                    0.0,
                    min(
                        1.0,
                        result["text_similarity"]
                    )
                ) * 100,
                1
            ),


            "color_match": None,

            "location_match": None,

            "time_match": None
        })


    return {
        "status": "success",
        "results": frontend_results
    }


if __name__ == "__main__":
    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )