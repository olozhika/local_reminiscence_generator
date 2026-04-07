import json
import re
import logging
from typing import List, Optional
from models import MultiDaySummary, DailySummary, Event, MemoryNode

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

    async def generate_multi_day_summary(self, conversation_text: str, existing_nodes_context: str = "") -> MultiDaySummary | None:
        # 核心逻辑：让 AI 识别日期并按天总结
        schema_dict = {
            "type": "object",
            "properties": {
                "summaries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
                            "events": {
                                "type": "array",
                                "items": Event.model_json_schema()
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
                        "required": ["date", "events", "nodes", "deleted_nodes"]
                    }
                }
            },
            "required": ["summaries"]
        }
        
        schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2).replace("{ai_name}", self.ai_name)
        
        system_prompt = f'''请根据提供的聊天记录总结事件并提取/更新记忆节点。
【输入说明】
输入可能包含多个聊天片段，每个片段以“--- 文件: 文件名 ---”开头。
【重要任务】
1. 识别日期：请从对话内容的时间戳中推断出每段对话发生的具体日期（格式 YYYY-MM-DD）。
2. 按天总结：为每个识别出的日期生成一个总结对象。
3. 提取事件 (events)：将对话整合为完整事件。
4. 提取节点 (nodes)：提取重要实体（人物、地点、概念）。
5. 继承与演进：参考已知节点背景，更新状态。

必须严格按照以下 JSON Schema 输出 JSON 数据：
{schema_str}

{f"【已知记忆节点背景】\n{existing_nodes_context}" if existing_nodes_context else ""}
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
            
            return MultiDaySummary(**data)
        except Exception as e:
            logger.error(f"多日总结失败: {e}")
            return None
