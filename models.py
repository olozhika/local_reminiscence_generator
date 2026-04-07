from pydantic import BaseModel, Field
from typing import List, Optional

class Event(BaseModel):
    event_id: str = Field(description="唯一ID，格式 evt_YYYYMMDD_序号")
    narrative: str = Field(description="事件完整叙述")
    emotion: str = Field(description="AI自己的情绪反应")
    importance: int = Field(ge=1, le=10, description="重要性 1-10")
    emotional_intensity: int = Field(ge=1, le=10, description="情绪强度 1-10")
    tags: List[str] = Field(description="标签数组")

class MemoryNode(BaseModel):
    name: str = Field(description="节点名称")
    type: str = Field(description="节点类型")
    description: str = Field(description="描述")

class DailySummary(BaseModel):
    date: str = Field(description="日期 YYYY-MM-DD")
    events: List[Event] = Field(description="事件列表")
    nodes: List[MemoryNode] = Field(description="记忆节点")
    deleted_nodes: List[str] = Field(default_factory=list, description="删除的节点")

class MultiDaySummary(BaseModel):
    summaries: List[DailySummary] = Field(description="按日期分类的总结列表")
