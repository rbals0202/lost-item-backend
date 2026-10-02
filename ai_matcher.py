from pathlib import Path

import clip
import torch
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def _validate_description(description, field="description"):
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{field}: 비어 있지 않은 설명 문자열이 필요합니다.")

    return description


def _validate_image_path(image_path, field="image_path"):
    if not isinstance(image_path, (str, Path)) or not str(image_path).strip():
        raise ValueError(
            f"{field}: 이미지 경로가 필요합니다. 현재는 사진이 있는 신고만 지원합니다."
        )

    path = Path(image_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{field}: 이미지 파일을 찾을 수 없습니다: {path}"
        )

    return path


def embed_image(image_path):
    path = _validate_image_path(image_path)

    try:
        with Image.open(path) as image:
            tensor = preprocess(
                image.convert("RGB")
            ).unsqueeze(0).to(device)

    except (OSError, ValueError) as exc:
        raise ValueError(
            f"이미지 파일을 읽을 수 없습니다: {path} ({exc})"
        ) from exc

    with torch.no_grad():
        feature = model.encode_image(tensor).float()
        return feature / feature.norm(dim=-1, keepdim=True)


def embed_text(description):
    description = _validate_description(description)

    tokens = clip.tokenize(
        [description],
        truncate=True
    ).to(device)

    with torch.no_grad():
        feature = model.encode_text(tokens).float()
        return feature / feature.norm(dim=-1, keepdim=True)


def match_lost_item(lost_item, candidates, top_k=5):

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k: 1 이상의 정수가 필요합니다.")

    if not isinstance(lost_item, dict):
        raise ValueError("lost_item: 분실물 정보 dict가 필요합니다.")

    if not isinstance(candidates, list):
        raise ValueError(
            "candidates: 습득물 정보 dict의 list가 필요합니다."
        )

    query_path = _validate_image_path(
        lost_item.get("image_path"),
        "lost_item.image_path"
    )

    query_description = _validate_description(
        lost_item.get("description"),
        "lost_item.description"
    )

    validated = []

    for index, candidate in enumerate(candidates):

        field = f"candidates[{index}]"

        if not isinstance(candidate, dict):
            raise ValueError(
                f"{field}: 습득물 정보 dict가 필요합니다."
            )

        item_id = candidate.get("item_id")

        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(
                f"{field}.item_id: 비어 있지 않은 문자열이 필요합니다."
            )

        field += f" (item_id={item_id})"

        description = _validate_description(
            candidate.get("description"),
            f"{field}.description"
        )

        images = candidate.get("images")

        if not isinstance(images, list) or not images:
            raise ValueError(
                f"{field}.images: 한 장 이상의 이미지 경로 list가 필요합니다."
            )

        paths = [
            _validate_image_path(
                path,
                f"{field}.images[{i}]"
            )
            for i, path in enumerate(images)
        ]

        validated.append(
            (
                item_id,
                description,
                paths
            )
        )

    if not validated:
        return []

    image_cache = {}
    text_cache = {}

    def image_embedding(path):
        if path not in image_cache:
            image_cache[path] = embed_image(path)

        return image_cache[path]

    def text_embedding(description):
        if description not in text_cache:
            text_cache[description] = embed_text(description)

        return text_cache[description]

    query_image = image_embedding(query_path)
    query_text = text_embedding(query_description)

    results = []

    for item_id, description, paths in validated:

        image_similarity = max(
            float(
                (
                    query_image
                    @ image_embedding(path).T
                ).item()
            )
            for path in paths
        )

        text_similarity = float(
            (
                query_text
                @ text_embedding(description).T
            ).item()
        )

        matching_score = (
            0.5 * image_similarity
            + 0.5 * text_similarity
        )

        results.append(
            {
                "item_id": item_id,
                "matching_score": matching_score,
                "image_similarity": image_similarity,
                "text_similarity": text_similarity
            }
        )

    return sorted(
        results,
        key=lambda row: row["matching_score"],
        reverse=True
    )[:top_k]