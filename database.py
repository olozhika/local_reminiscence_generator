import sqlite3
import json
import logging
from contextlib import contextmanager
from typing import List
from models import BatchSummary, MemoryNode

logger = logging.getLogger(__name__)

class MemoryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库，完全兼容 Astrbot 插件的表结构"""
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
            
            # 5. nodes 表 (支持别名和关联事件)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    name TEXT PRIMARY KEY,
                    type TEXT,
                    description TEXT,
                    aliases TEXT DEFAULT '[]',
                    related_events TEXT DEFAULT '[]',
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

    def _get_unique_event_id(self, cursor, event_id):
        """
        检查事件ID是否重复，如果重复则自动修改序号。
        例如: evt_20180101_001 -> evt_20180101_002
        """
        original_id = event_id
        counter = 1
        
        while True:
            cursor.execute("SELECT COUNT(*) FROM events WHERE event_id = ?", (event_id,))
            if cursor.fetchone()[0] == 0:
                return event_id
            
            # ID重复，修改序号
            counter += 1
            # 解析原始ID，修改序号部分
            parts = original_id.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                event_id = f"{parts[0]}_{counter:03d}"
            else:
                event_id = f"{original_id}_{counter}"

    def insert_summary(self, summary: BatchSummary):
        """插入总结数据，节点采用追加策略，事件ID自动去重"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 先插入事件（因为节点的related_events需要引用事件ID）
            for event in summary.events:
                # 检查并确保事件ID唯一
                unique_event_id = self._get_unique_event_id(cursor, event.event_id)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO events 
                    (event_id, date, narrative, emotion, importance, emotional_intensity, reflection)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_event_id,
                    event.date,  # 使用事件自己的日期
                    event.narrative,
                    event.emotion,
                    event.importance,
                    event.emotional_intensity,
                    event.reflection if event.reflection != "无" else None
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
                            (unique_event_id, tag_id)
                        )

            # 追加记忆节点（合并而非覆写）
            for node in summary.nodes:
                # 检查节点是否已存在
                cursor.execute("SELECT * FROM nodes WHERE name = ?", (node.name,))
                existing = cursor.fetchone()
                
                if existing:
                    # 合并描述（追加新内容）
                    old_desc = existing['description'] or ''
                    new_desc = node.description
                    # 如果新描述不是旧描述的子集，则追加
                    if new_desc and new_desc not in old_desc:
                        merged_desc = f"{old_desc}\n{new_desc}" if old_desc else new_desc
                    else:
                        merged_desc = old_desc
                    
                    # 合并别名（去重）
                    old_aliases = json.loads(existing['aliases'] or '[]')
                    new_aliases = node.aliases
                    merged_aliases = list(set(old_aliases + new_aliases))
                    
                    # 合并关联事件（去重）
                    old_events = json.loads(existing['related_events'] or '[]')
                    new_events = node.related_events
                    merged_events = list(set(old_events + new_events))
                    
                    cursor.execute("""
                        UPDATE nodes SET 
                            type = ?,
                            description = ?,
                            aliases = ?,
                            related_events = ?,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE name = ?
                    """, (node.type, merged_desc, json.dumps(merged_aliases, ensure_ascii=False), 
                          json.dumps(merged_events, ensure_ascii=False), node.name))
                else:
                    # 新节点，直接插入
                    cursor.execute("""
                        INSERT INTO nodes (name, type, description, aliases, related_events, last_updated)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (node.name, node.type, node.description, 
                          json.dumps(node.aliases, ensure_ascii=False),
                          json.dumps(node.related_events, ensure_ascii=False)))

            # 删除冗余节点
            if summary.deleted_nodes:
                placeholders = ','.join(['?'] * len(summary.deleted_nodes))
                cursor.execute(f"DELETE FROM nodes WHERE name IN ({placeholders})", summary.deleted_nodes)

            conn.commit()

    def get_all_nodes(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            return [dict(row) for row in cursor.fetchall()]

    def get_events_count(self):
        """获取事件总数"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]

    def deduplicate_nodes(self):
        """节点去重：合并同名节点的描述、别名和关联事件"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            nodes = [dict(row) for row in cursor.fetchall()]
            
            # 按名称分组（理论上不应该有重复，因为name是PRIMARY KEY）
            # 但我们可以处理描述中的重复内容
            for node in nodes:
                desc = node['description'] or ''
                aliases = json.loads(node['aliases'] or '[]')
                events = json.loads(node['related_events'] or '[]')
                
                # 去重别名和事件
                unique_aliases = list(set(aliases))
                unique_events = list(set(events))
                
                # 如果有变化，更新
                if len(unique_aliases) != len(aliases) or len(unique_events) != len(events):
                    cursor.execute("""
                        UPDATE nodes SET 
                            aliases = ?,
                            related_events = ?
                        WHERE name = ?
                    """, (json.dumps(unique_aliases, ensure_ascii=False),
                          json.dumps(unique_events, ensure_ascii=False),
                          node['name']))
            
            conn.commit()
            logger.info(f"节点去重完成，处理了 {len(nodes)} 个节点")

    def get_all_nodes_list(self):
        """获取所有节点列表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nodes")
            return [dict(row) for row in cursor.fetchall()]

    def update_node(self, name: str, type: str = None, description: str = None, 
                    aliases: List[str] = None, related_events: List[str] = None):
        """更新单个节点"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 构建更新语句
            updates = []
            params = []
            if type is not None:
                updates.append("type = ?")
                params.append(type)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if aliases is not None:
                updates.append("aliases = ?")
                params.append(json.dumps(aliases, ensure_ascii=False))
            if related_events is not None:
                updates.append("related_events = ?")
                params.append(json.dumps(related_events, ensure_ascii=False))
            
            if updates:
                updates.append("last_updated = CURRENT_TIMESTAMP")
                params.append(name)
                sql = f"UPDATE nodes SET {', '.join(updates)} WHERE name = ?"
                cursor.execute(sql, params)
                conn.commit()

    def delete_nodes(self, node_names: List[str]):
        """批量删除节点"""
        if not node_names:
            return
        with self._get_conn() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(node_names))
            cursor.execute(f"DELETE FROM nodes WHERE name IN ({placeholders})", node_names)
            conn.commit()
