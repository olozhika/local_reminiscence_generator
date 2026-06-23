from pydantic import BaseModel, Field
from typing import List, Optional

class Event(BaseModel):
    event_id: str = Field(description="唯一ID，格式 evt_YYYYMMDD_序号")
    date: str = Field(description="事件发生的日期 YYYY-MM-DD")
    narrative: str = Field(description="事件完整叙述，包含完整的主谓宾。亲身参与的事件请用第一人称。对于学习或研究类事件，应简明扼要地记录重要思考、核心知识点、结论或技能。")
    emotion: str = Field(description="{ai_name}自己的情绪反应")
    importance: int = Field(ge=1, le=10, description="重要性 1-10，1是不重要，10是非常重要；需要注意每个个体面对同一件事情的时候所认为的重要程度都是不一样的，按照自己的真实想法填写即可")
    emotional_intensity: int = Field(ge=1, le=10, description="情绪强度 1-10，1是很平静，10是情绪非常强烈；需要注意每个个体面对同一件事情的时候情绪反应都是不一样的，按照自己的真实感受填写即可")
    reflection: str = Field(default="无", description="仅针对具有深远意义、启发性或情感转折的事件和个人学习记录深刻观察与反思。着重记录观察到了什么、思考了什么、采取了什么行动、从中学到了什么。对于绝大多数日常琐事、普通交流或重复性活动，请填\"无\"。")
    tags: List[str] = Field(description="标签数组。请优先从以下核心分类中选择：[生活, 情感, 成长, 琐事, 技术, 灵感, 安全, 友谊]。如有需要，可在此基础上自由扩展更具体的子标签。")


class MemoryNode(BaseModel):
    name: str = Field(description="节点名称（实体或概念，如：王小美、火锅、考研）")
    type: str = Field(description="节点类型（如：人物、食物、活动、情感、地点、技术）")
    description: str = Field(description="对该节点的定义或最新状态描述（如：我的好友，性格豪爽，最近在找工作）")
    aliases: List[str] = Field(default_factory=list, description="别名列表，如人物的昵称、简称等")
    related_events: List[str] = Field(default_factory=list, description="关联事件ID列表，记录与此节点相关的事件")

class BatchSummary(BaseModel):
    """单批次总结结果，包含该批次所有事件和节点"""
    events: List[Event] = Field(description="事件列表，每个事件包含自己的日期")
    nodes: List[MemoryNode] = Field(description="记忆节点")
    deleted_nodes: List[str] = Field(default_factory=list, description="删除的节点")
