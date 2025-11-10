import aiosqlite as aqlite
import json
from typing import Optional, Dict, Any

DB_FILE = "hair_color_cache.db"

async def init_db():
    """Initializes the cache database and creates the table if it doesn't exist."""
    async with aqlite.connect(DB_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                image_hash TEXT PRIMARY KEY,
                analysis_result TEXT NOT NULL
            )
        """)
        await db.commit()

async def get_cached_result(image_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieves an analysis result from the cache using the image hash."""
    try:
        async with aqlite.connect(DB_FILE) as db:
            cursor = await db.execute("SELECT analysis_result FROM image_cache WHERE image_hash = ?", (image_hash,))
            result = await cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None
    except Exception as e:
        print(f"Error getting cached result: {e}")
        return None

async def cache_result(image_hash: str, analysis_result: Dict[str, Any]):
    """Caches a new analysis result in the database."""
    try:
        async with aqlite.connect(DB_FILE) as db:
            await db.execute(
                "INSERT OR REPLACE INTO image_cache (image_hash, analysis_result) VALUES (?, ?)",
                (image_hash, json.dumps(analysis_result))
            )
            await db.commit()
    except Exception as e:
        print(f"Error caching result: {e}")

async def clear_cache():
    """Deletes all records from the image_cache table."""
    async with aqlite.connect(DB_FILE) as db:
        await db.execute("DELETE FROM image_cache")
        await db.commit()