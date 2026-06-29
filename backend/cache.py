# -*- coding: utf-8 -*-
"""캡션 캐시 — 같은 이미지는 캡션을 한 번만 뽑아 재사용.
3주차: 메모리 dict. (지속성 필요 시 SQLite, 운영 로깅 DB는 9주차)
캐시 키 = (image_id, model, prompt_version)  ← 모델/프롬프트 다르면 캡션도 다름
"""
import hashlib

_cache = {}  # {(image_id, model, prompt_version): caption}


def image_id(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def get(image_bytes, model, prompt_version):
    return _cache.get((image_id(image_bytes), model, prompt_version))


def put(image_bytes, model, prompt_version, caption):
    _cache[(image_id(image_bytes), model, prompt_version)] = caption
