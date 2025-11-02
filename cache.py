import aiosqlite as aqlite 
import json
from typing import Optional, Tuple

DB_FILE = "hair_color_cache.db"

async def init_db():
    async with aqlite.connect(DB_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                image_hash TEXT PRIMARY KEY,
                dominant_color_rgb TEXT,
                dominant_color_hex TEXT
            )
        """)
        await db.commit()

async def get_cached_result(image_hash: str) -> Optional[Tuple[str, str]]:
    async with aqlite.connect(DB_FILE) as db:
        cursor = await db.execute("SELECT dominant_color_rgb, dominant_color_hex FROM image_cache WHERE image_hash = ?", (image_hash,))
        result = await cursor.fetchone()
        if result:
            return json.loads(result[0]), result[1]
        return None

async def cache_result(image_hash: str, dominant_color_rgb: Tuple[int, int, int], dominant_color_hex: str):
    async with aqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT OR REPLACE INTO image_cache (image_hash, dominant_color_rgb, dominant_color_hex) VALUES (?, ?, ?)",
            (image_hash, json.dumps(dominant_color_rgb), dominant_color_hex)
        )
        await db.commit()