import sqlite3
from contextlib import contextmanager
from typing import List
from models import DailySummary, MemoryNode

class MemoryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库，完全兼容 Astrbot APLR 插件的表结构"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 1. events 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    narrative TEXT NOT NULL,
                    emotion TEXT,
                    importance INTEGER CHECK(importance BETWEEN 1 AND 10),
                    emotional_intensity INTEGER CHECK(emotional_intensity BETWEEN 1 AND 10),
                    reflection TEXT,
                    reinforcement_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")

            # 2. tags 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """)

            # 3. event_tags 表 (关联表)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_tags (
                    event_id TEXT,
                    tag_id INTEGER,
                    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                    PRIMARY KEY (event_id, tag_id)
                )
            """)

            # 4. daily_reflections 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_reflections (
                    date TEXT PRIMARY KEY,
                    reflection TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 5. nodes 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    name TEXT PRIMARY KEY,
                    type TEXT,
                    description TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 6. event_relations 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_relations (
                    source_event_id TEXT,
                    target_event_id TEXT,
                    relation_type TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_event_id, target_event_id, relation_type),
                    FOREIGN KEY(source_event_id) REFERENCES events(event_id) ON DELETE CASCADE,
                    FOREIGN KEY(target_event_id) REFERENCES events(event_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def insert_summary(self, summary: DailySummary):
        """插入总结数据，兼容 Astrbot 逻辑"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 插入/更新记忆节点
            for node in summary.nodes:
                cursor.execute("""
                    INSERT INTO nodes (name, type, description, last_updated)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(name) DO UPDATE SET
                        type = excluded.type,
                        description = excluded.description,
                        last_updated = CURRENT_TIMESTAMP
                """, (node.name, node.type, node.description))

            # 删除冗余节点
            if summary.deleted_nodes:
                placeholders = ','.join(['?'] * len(summary.deleted_nodes))
                cursor.execute(f"DELETE FROM nodes WHERE name IN ({placeholders})", summary.deleted_nodes)

            # 插入事件
            for event in summary.events:
                cursor.execute("""
                    INSERT OR REPLACE INTO events 
                    (event_id, date, narrative, emotion, importance, emotional_intensity, reflection)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    summary.date,
                    event.narrative,
                    event.emotion,
                    event.importance,
                    event.emotional_intensity,
                    None # reflection 不再由 AI 填写
                ))
                
                # 处理标签
                for tag_name in event.tags:
                    cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
                    tag_row = cursor.fetchone()
                    if tag_row:
                        tag_id = tag_row['id']
                        cursor.execute(
                            "INSERT OR IGNORE INTO event_tags (event_id, tag_id) VALUES (?, ?)",
                            (event.event_id, tag_id)
                        )
            conn.commit()

    def get_all_nodes(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            return [dict(row) for row in cursor.fetchall()]
