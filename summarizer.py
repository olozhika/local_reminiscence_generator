import json
import re
import logging
from typing import List, Optional, Dict, Any
from models import BatchSummary, Event, MemoryNode

logger = logging.getLogger(__name__)

class DailySummarizer:
    def __init__(self, llm_generate_func, ai_name: str = "Lanya", base_system_prompt: str = ""):
        self.llm_generate = llm_generate_func
        self.ai_name = ai_name
        self.base_system_prompt = base_system_prompt.strip()

    def _extract_json(self, text: str) -> str:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        obj_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if obj_match:
            return obj_match.group(1).strip()
        return text.strip()

    async def optimize_nodes_batch(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
        """
        优化一批节点：合并重复描述、去重别名、形成最终认知。
        返回优化后的节点列表。
        """
        if not nodes:
            return []
        
        schema = {
            "type": "object",
            "properties": {
                "optimized_nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "节点名称（保持原名）"},
                            "type": {"type": "string", "description": "节点类型"},
                            "description": {"type": "string", "description": "优化后的描述，去除重复，形成对这个概念的最终认知"},
                            "aliases": {"type": "array", "items": {"type": "string"}, "description": "去重后的别名列表"},
                            "related_events": {"type": "array", "items": {"type": "string"}, "description": "关联事件ID列表"}
                        },
                        "required": ["name", "type", "description", "aliases", "related_events"]
                    }
                },
                "nodes_to_delete": {"type": "array", "items": {"type": "string"}, "description": "应该删除的冗余节点名称"}
            },
            "required": ["optimized_nodes", "nodes_to_delete"]
        }
        
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        # 准备节点信息
        nodes_info = []
        for node in nodes:
            aliases = json.loads(node.get('aliases', '[]') or '[]')
            events = json.loads(node.get('related_events', '[]') or '[]')
            nodes_info.append({
                "name": node['name'],
                "type": node['type'] or '未知',
                "description": node['description'] or '',
                "aliases": aliases,
                "related_events": events
            })
        
        system_prompt = f'''请对以下记忆节点进行优化整理。
【任务】
1. 合并重复：如果多个节点描述的是同一个概念，请合并它们的信息。
2. 去重别名：去除重复的别名。
3. 优化描述：去除描述中的重复内容，形成对这个概念的清晰、完整的最终认知。
4. 识别冗余：如果某些节点完全被其他节点包含，标记为待删除。

【描述风格要求】
- 使用第一人称描述（如"我的好友"、"我常去的地方"、"我感兴趣的课题"）
- 保持简洁自然，像在写个人笔记

【输出要求】
必须严格按照以下 JSON Schema 输出：
{schema_str}
'''

        try:
            llm_resp = await self.llm_generate(
                prompt=f"需要优化的记忆节点：\n\n{json.dumps(nodes_info, ensure_ascii=False, indent=2)}",
                system_prompt=system_prompt,
            )
            content = self._extract_json(llm_resp)
            data = json.loads(content)
            return data
        except Exception as e:
            logger.error(f"节点优化失败: {e}")
            return None

    async def generate_batch_summary(self, conversation_text: str, existing_nodes_context: str = "") -> BatchSummary | None:
        """
        生成单批次总结，返回一个包含所有事件和节点的总结对象。
        """
        # 生成Event schema并替换{ai_name}占位符
        event_schema = Event.model_json_schema()
        event_schema_str = json.dumps(event_schema, ensure_ascii=False).replace("{ai_name}", self.ai_name)
        event_schema = json.loads(event_schema_str)
        
        schema_dict = {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": event_schema,
                    "description": "事件列表，每个事件包含自己的日期"
                },
                "nodes": {
                    "type": "array",
                    "items": MemoryNode.model_json_schema()
                },
                "deleted_nodes": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["events", "nodes", "deleted_nodes"]
        }
        
        schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2)
        
        system_prompt = f'''请根据提供的聊天记录，以第一人称视角总结事件并提取/更新记忆节点。

【输入说明】
- 输入可能包含多个聊天片段，每个片段以"--- 文件: 文件名 ---"开头
- 请仔细识别哪些是自己说的话，哪些是对方说的话
- 聊天记录可能跨越多天，请为每个事件标注正确的日期
- 部分聊天记录可能没头没尾，比如“XXX说今天谢谢你了，我说哈哈不客气”“XXX发送了图片，我也发送了图片”，这种不用总结成事件

【叙事风格要求】
请参考自己当时的表达方式自然、简洁地叙述事件


【重要任务】
1. 提取事件 (events)：
   - date：事件发生的日期（YYYY-MM-DD格式）
   - narrative：用第一人称叙述，像写日记一样记录自己经历了什么
   - emotion：描述自己当时的情绪（如"开心""好奇""有点烦""兴奋"）
   - importance：对自己的重要程度（1-10）
   - emotional_intensity：自己感受到的情绪强度（1-10）
   - reflection：仅对具有深远意义、启发性或情感转折的事件填写反思，记录观察、思考、行动和收获。绝大多数日常琐事请填"无"
   - tags：优先从 [生活, 情感, 琐事, 兴趣, 学习, 家庭, 友谊, 出游, 创作, 科研] 中选择，可自由扩展
2. 提取节点 (nodes)：
   - 使用第一人称
   - 提取重要的概念，比如人物、地点、组织、活动
   - 为节点设置别名（如昵称、简称、外号）
   - 记录关联事件ID
3. 继承与演进：提取时间和节点时可参考已知节点背景，记录节点时只提供增量及纠正信息

必须严格按照以下 JSON Schema 输出：
{schema_str}

{f"【已知记忆节点背景】" + chr(10) + existing_nodes_context if existing_nodes_context else ""}
'''
        if self.base_system_prompt:
            system_prompt = self.base_system_prompt + "\n\n" + system_prompt

        try:
            llm_resp = await self.llm_generate(
                prompt=f"聊天记录（可能包含多日内容）：\n\n{conversation_text}",
                system_prompt=system_prompt,
            )
            content = self._extract_json(llm_resp)
            data = json.loads(content)
            
            return BatchSummary(**data)
        except Exception as e:
            logger.error(f"批次总结失败: {e}")
            return None
