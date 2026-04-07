import json
import asyncio
import logging
import os
import re
from datetime import datetime
from openai import AsyncOpenAI
from database import MemoryDB
from summarizer import DailySummarizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None

def parse_chat_content(content):
    """
    解析文本格式聊天记录。
    格式: YYYY-MM-DD HH:MM:SS 姓名\n内容
    返回: List[dict] 包含 timestamp, sender, text
    """
    messages = []
    # 匹配时间戳和姓名行
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(.*)$')
    
    lines = content.splitlines()
    current_msg = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = pattern.match(line)
        if match:
            if current_msg:
                messages.append(current_msg)
            
            ts_str = match.group(1)
            sender = match.group(2)
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except:
                ts = datetime.now() # 降级处理
                
            current_msg = {
                "timestamp": ts,
                "timestamp_str": ts_str,
                "sender": sender,
                "text": ""
            }
        else:
            if current_msg:
                if current_msg["text"]:
                    current_msg["text"] += "\n" + line
                else:
                    current_msg["text"] = line
    
    if current_msg:
        messages.append(current_msg)
        
    return messages

def get_atomic_blocks(messages, time_gap_hours):
    """
    将消息列表按时间间隔切分为原子块。
    """
    if not messages: return []
    
    blocks = []
    current_block = [messages[0]]
    
    for i in range(1, len(messages)):
        gap = (messages[i]["timestamp"] - messages[i-1]["timestamp"]).total_seconds() / 3600
        if gap >= time_gap_hours:
            blocks.append(current_block)
            current_block = [messages[i]]
        else:
            current_block.append(messages[i])
            
    blocks.append(current_block)
    return blocks

def format_block(block, filename):
    """将消息块格式化为字符串"""
    header = f"--- 文件: {filename} ---\n"
    body = ""
    for msg in block:
        body += f"{msg['timestamp_str']} {msg['sender']}\n{msg['text']}\n\n"
    return header + body

def filter_by_date_range(input_dir, start_date_str, end_date_str):
    """
    根据日期范围过滤聊天记录。
    """
    if not os.path.exists(input_dir):
        return

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        # 截至日期包含当天
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    except Exception as e:
        logger.error(f"日期格式错误: {e}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            path = os.path.join(input_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                
                messages = parse_chat_content(content)
                if not messages:
                    continue

                # 过滤消息
                filtered_messages = [msg for msg in messages if start_date <= msg["timestamp"] <= end_date]
                
                if len(filtered_messages) == len(messages):
                    continue # 无需修改

                new_content = ""
                for msg in filtered_messages:
                    new_content += f"{msg['timestamp_str']} {msg['sender']}\n{msg['text']}\n\n"
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.info(f"已完成日期过滤: {filename} (保留了 {len(filtered_messages)}/{len(messages)} 条消息)")
                
            except Exception as e:
                logger.error(f"过滤文件 {filename} 失败: {e}")

def preprocess_group_chats(input_dir, target_username, context_lines):
    """
    预处理群聊文件：保留目标用户发言及其前后的上下文。
    """
    if not os.path.exists(input_dir):
        return

    for filename in os.listdir(input_dir):
        if "群聊" in filename and filename.endswith(".txt"):
            path = os.path.join(input_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                
                messages = parse_chat_content(content)
                if not messages:
                    continue

                # 找出目标用户的发言索引
                target_indices = [i for i, msg in enumerate(messages) if msg["sender"] == target_username]
                
                if not target_indices:
                    # 如果文件中没有目标用户的发言，则清空文件或保留为空（根据需求，这里选择清空以减少干扰）
                    logger.info(f"文件 {filename} 中未找目标用户 {target_username}，已清空。")
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write("")
                    continue

                # 确定要保留的索引集合
                to_keep = set()
                for idx in target_indices:
                    start = max(0, idx - context_lines)
                    end = min(len(messages), idx + context_lines + 1)
                    for i in range(start, end):
                        to_keep.add(i)
                
                sorted_indices = sorted(list(to_keep))
                
                # 重构内容
                new_content = ""
                last_idx = -1
                for idx in sorted_indices:
                    # 如果不连续，添加分割线
                    if last_idx != -1 and idx != last_idx + 1:
                        new_content += "\n... (已省略无关对话) ...\n\n"
                    
                    msg = messages[idx]
                    new_content += f"{msg['timestamp_str']} {msg['sender']}\n{msg['text']}\n\n"
                    last_idx = idx
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.info(f"已完成群聊预处理: {filename} (保留了 {len(sorted_indices)}/{len(messages)} 条消息)")
                
            except Exception as e:
                logger.error(f"预处理文件 {filename} 失败: {e}")

async def main():
    # 1. 加载配置
    config = load_config()
    if not config:
        return

    api_cfg = config["api"]
    prompt_cfg = config["prompts"]
    file_cfg = config["files"]

    # 2. 预处理：日期范围过滤
    logger.info("开始日期范围过滤...")
    filter_by_date_range(
        file_cfg["input_dir"],
        file_cfg.get("start_date", "2000-01-01"),
        file_cfg.get("end_date", "2099-12-31")
    )

    # 3. 预处理：群聊文件
    logger.info("开始群聊预处理...")
    preprocess_group_chats(
        file_cfg["input_dir"], 
        file_cfg.get("target_username", ""), 
        file_cfg.get("context_lines", 10)
    )

    # 4. 初始化数据库
    db = MemoryDB(file_cfg["output_db"])
    
    # 5. 扫描并解析所有聊天文件
    input_dir = file_cfg["input_dir"]
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return

    all_blocks = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            path = os.path.join(input_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8-sig') as f: # 处理可能的 BOM
                    content = f.read()
                    messages = parse_chat_content(content)
                    blocks = get_atomic_blocks(messages, file_cfg["time_gap_hours"])
                    for b in blocks:
                        all_blocks.append({"filename": filename, "messages": b})
            except Exception as e:
                logger.error(f"读取文件 {filename} 失败: {e}")

    if not all_blocks:
        logger.info("未找到有效的聊天记录。")
        return

    # 6. 分批逻辑
    batches = []
    current_batch_text = ""
    max_bytes = file_cfg["batch_size_kb"] * 1024

    for block_info in all_blocks:
        block_text = format_block(block_info["messages"], block_info["filename"])
        block_size = len(block_text.encode('utf-8'))
        
        # 如果当前块本身就超过限制，且当前 batch 为空，则强制放入一个 batch
        if block_size > max_bytes and not current_batch_text:
            batches.append(block_text)
            continue
            
        if len(current_batch_text.encode('utf-8')) + block_size > max_bytes:
            batches.append(current_batch_text)
            current_batch_text = block_text
        else:
            current_batch_text += block_text

    if current_batch_text:
        batches.append(current_batch_text)

    # 7. 估计 Token 数并询问用户
    total_chars = sum(len(b) for b in batches)
    # 粗略估计：中文/混合文本 1个字符约 0.6~1.5 个 token (取决于模型和分词)
    # 这里给出一个保守的范围估计
    min_tokens = int(total_chars * 0.8)
    max_tokens = int(total_chars * 1.5)
    
    print("\n" + "="*50)
    print(f"待处理文本总长度: {total_chars} 字符")
    print(f"预计消耗 Token 数 (输入): 约 {min_tokens} ~ {max_tokens}")
    print(f"分批数量: {len(batches)} 批")
    print("="*50)
    
    user_input = input("是否确定开始总结？(y/n): ").strip().lower()
    if user_input != 'y':
        print("已取消处理。")
        return

    # 8. 初始化 LLM 客户端
    client = AsyncOpenAI(api_key=api_cfg["api_key"], base_url=api_cfg["base_url"])

    async def llm_generate(prompt, system_prompt):
        response = await client.chat.completions.create(
            model=api_cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    # 9. 执行总结
    summarizer = DailySummarizer(
        llm_generate, 
        ai_name=prompt_cfg["ai_name"], 
        base_system_prompt=prompt_cfg["system_prompt"]
    )
    
    logger.info(f"开始处理，共分为 {len(batches)} 个批次。")
    
    for i, batch_text in enumerate(batches):
        logger.info(f"正在处理批次 {i+1}/{len(batches)} (大小: {len(batch_text.encode('utf-8'))/1024:.2f} KB)...")
        
        # 每次总结前重新获取最新节点背景
        existing_nodes = db.get_all_nodes()
        nodes_context = "\n".join([f"- {n['name']} ({n['type']}): {n['description']}" for n in existing_nodes])

        multi_day_summary = await summarizer.generate_multi_day_summary(batch_text, nodes_context)

        if multi_day_summary and multi_day_summary.summaries:
            for summary in multi_day_summary.summaries:
                logger.info(f"正在处理日期: {summary.date}")
                db.insert_summary(summary)
        else:
            logger.error(f"批次 {i+1} 总结生成失败。")

    logger.info(f"所有批次处理完成。数据库: {file_cfg['output_db']}")

if __name__ == "__main__":
    asyncio.run(main())
