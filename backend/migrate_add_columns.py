import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_url():
    url = os.getenv('DATABASE_URL', '')
    # Convert postgresql+asyncpg:// to postgresql:// for asyncpg
    if url.startswith('postgresql+asyncpg://'):
        url = url.replace('postgresql+asyncpg://', 'postgresql://')
    return url

async def add_columns():
    conn = await asyncpg.connect(get_db_url())
    
    migrations = [
        'ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS name VARCHAR(200)',
        'ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS is_shared BOOLEAN DEFAULT FALSE',
        'ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS share_token VARCHAR(100) UNIQUE',
        'ALTER TABLE ml_sessions ADD COLUMN IF NOT EXISTS shared_at TIMESTAMP WITHOUT TIME ZONE',
    ]
    
    for migration in migrations:
        try:
            await conn.execute(migration)
            print(f'OK: {migration[:60]}')
        except Exception as e:
            print(f'FAIL: {migration[:60]} - {e}')
    
    await conn.close()
    print('Migration complete!')

if __name__ == '__main__':
    asyncio.run(add_columns())