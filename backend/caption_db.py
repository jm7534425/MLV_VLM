# -*- coding: utf-8 -*-
"""Caption DB (6주차 Phase 5) — 구조화 캡션을 SQLite에 영속 저장·재사용.
3주차 메모리 dict(cache.py) → 영속 DB로 승격. 서버 재시작해도 캡션 유지.
키 = (image_id, model, prompt_version). 값 = semantic{} + measured{} + flags[]."""
import os, json, sqlite3, hashlib
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "caption_db.sqlite")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS captions(
        image_id       TEXT,
        model          TEXT,
        prompt_version TEXT,
        objects        TEXT,   -- 조회 편의용 (semantic_json에도 포함)
        scene          TEXT,
        semantic_json  TEXT,   -- objects·body_parts·scene·style·spatial_layout·completeness
        measured_json  TEXT,   -- 코드 측정값
        flags_json     TEXT,   -- 교차검증 플래그
        created_at     TEXT,
        PRIMARY KEY (image_id, model, prompt_version))""")
    return c


def image_id(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def get(image_bytes, model, prompt_version):
    """저장된 구조화 캡션 반환 (없으면 None)."""
    c = _conn()
    row = c.execute(
        "SELECT semantic_json, measured_json, flags_json FROM captions "
        "WHERE image_id=? AND model=? AND prompt_version=?",
        (image_id(image_bytes), model, prompt_version)).fetchone()
    c.close()
    if not row:
        return None
    obj = json.loads(row[0])
    obj["measured"] = json.loads(row[1])
    obj["flags"] = json.loads(row[2])
    return obj


def put(image_bytes, model, prompt_version, cap_obj):
    """구조화 캡션 저장 (이미 있으면 갱신)."""
    semantic = {k: v for k, v in cap_obj.items() if k not in ("measured", "flags")}
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO captions VALUES (?,?,?,?,?,?,?,?,?)",
        (image_id(image_bytes), model, prompt_version,
         json.dumps(cap_obj.get("objects", []), ensure_ascii=False),
         cap_obj.get("scene", ""),
         json.dumps(semantic, ensure_ascii=False),
         json.dumps(cap_obj.get("measured", {}), ensure_ascii=False),
         json.dumps(cap_obj.get("flags", []), ensure_ascii=False),
         datetime.now(timezone.utc).isoformat()))
    c.commit()
    c.close()


def stats(limit=5):
    """DB 현황 (개수 + 최근 몇 개) — 데모·검증용."""
    c = _conn()
    n = c.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
    rows = c.execute(
        "SELECT image_id, model, prompt_version, scene, created_at "
        "FROM captions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return {"count": n, "recent": [
        {"image_id": r[0], "model": r[1], "prompt_version": r[2],
         "scene": r[3], "created_at": r[4]} for r in rows]}
