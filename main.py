import json
import asyncio
import logging
import os
import re
import shutil
import psutil
import random
import sys
from datetime import datetime
from openai import AsyncOpenAI
from database import MemoryDB
from summarizer import DailySummarizer

# ============ 用户配置 ============
AUTO_CONFIRM = False  # True: 跳过确认直接开始; False: 需要输入y确认
# ===================================

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# jieba分词配置
try:
    import jieba
    jieba.setLogLevel(logging.WARNING)  # 关闭jieba的日志输出
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    logger.warning("jieba未安装，将使用简单的字符串匹配。可通过 pip install jieba 安装。")

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None

def get_dynamic_concurrency(max_concurrent_config=None):
    """
    动态计算并发数。
    为了降低风控风险，默认并发数较低。
    """
    if max_concurrent_config and max_concurrent_config > 0:
        return max_concurrent_config
    
    # 获取CPU核心数
    cpu_count = psutil.cpu_count(logical=True) or 4
    
    # 为了降低风控风险，使用较低的并发数
    # 建议不超过3个并发
    concurrency = min(3, cpu_count)
    
    logger.info(f"CPU核心数: {cpu_count}, 并发数: {concurrency} (已限制以降低风控风险)")
    return concurrency

def parse_chat_content(content, source_type="auto"):
    """
    解析文本格式聊天记录，支持QQ和微信两种格式。
    QQ格式: YYYY-MM-DD HH:MM:SS 姓名\n内容
    微信格式: YYYY-MM-DD HH:MM:SS '姓名'\n内容
    返回: List[dict] 包含 timestamp, sender, text
    """
    messages = []
    
    # 跳过QQ记录的头部信息
    lines = content.splitlines()
    start_idx = 0
    
    # 检测是否为QQ记录格式（有"消息记录"头部）
    if lines and "消息记录" in lines[0]:
        # 跳过QQ头部，找到第一个时间戳行
        for i, line in enumerate(lines):
            if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line.strip()):
                start_idx = i
                break
    
    # 匹配时间戳和姓名行（支持带引号和不带引号的格式）
    # QQ格式: 2017-08-01 12:28:21 摇颺葳蕤
    # 微信格式: 2024-07-11 16:02:59 'Astrabbit'
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+[\'"]?(.*?)[\'"]?\s*$')
    
    current_msg = None
    
    for line in lines[start_idx:]:
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

def get_text_tokens(text):
    """对文本进行分词，返回token集合"""
    if HAS_JIEBA:
        # 使用jieba分词，过滤掉单字符和标点
        tokens = set()
        for word in jieba.cut(text):
            word = word.strip()
            if len(word) >= 2 and not re.match(r'^[\s\W]+$', word):
                tokens.add(word)
        return tokens
    else:
        # 简单的按字符和空格分割
        tokens = set()
        for word in re.split(r'[\s,，。！？.!?\n]+', text):
            if len(word) >= 2:
                tokens.add(word)
        return tokens

def match_relevant_nodes(nodes, batch_text, max_nodes):
    """
    根据聊天内容匹配相关节点。
    匹配策略：
    1. 节点名称或别名直接出现在聊天内容中
    2. 使用分词，聊天内容的token与节点名称/别名的token有交集
    """
    if not nodes:
        return []
    
    # 获取聊天内容的tokens
    batch_tokens = get_text_tokens(batch_text)
    
    scored_nodes = []
    for node in nodes:
        score = 0
        name = node.get('name', '')
        aliases = json.loads(node.get('aliases', '[]') or '[]')
        desc = node.get('description', '')
        
        # 所有可能的标识符
        all_names = [name] + aliases
        
        for identifier in all_names:
            # 1. 直接包含匹配（最高优先级）
            if identifier in batch_text:
                score += 100
                break
            # 2. 分词匹配
            identifier_tokens = get_text_tokens(identifier)
            if identifier_tokens & batch_tokens:  # 有交集
                score += 50
                break
        
        # 3. 描述内容的分词匹配（较低优先级）
        if score == 0 and desc:
            desc_tokens = get_text_tokens(desc)
            overlap = desc_tokens & batch_tokens
            if len(overlap) >= 2:  # 至少2个token重叠才算匹配
                score += len(overlap) * 5
        
        if score > 0:
            scored_nodes.append((score, node))
    
    # 按分数降序排序，取前max_nodes个
    scored_nodes.sort(key=lambda x: x[0], reverse=True)
    return [node for _, node in scored_nodes[:max_nodes]]

def format_nodes_context(nodes):
    """格式化节点上下文字符串"""
    parts = []
    for n in nodes:
        aliases = json.loads(n.get('aliases', '[]') or '[]')
        aliases_str = f" (别名: {', '.join(aliases)})" if aliases else ""
        parts.append(f"- {n['name']}{aliases_str} ({n['type']}): {n['description']}")
    return "\n".join(parts)

def get_all_txt_files(input_dir):
    """
    递归获取目录下所有txt文件，返回 (文件路径, 来源类型) 列表。
    来源类型: "qq_private", "qq_group", "wechat"
    """
    txt_files = []
    if not os.path.exists(input_dir):
        return txt_files
    
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".txt"):
                path = os.path.join(root, filename)
                # 根据目录名判断来源类型
                rel_path = os.path.relpath(path, input_dir)
                if "QQ私聊" in rel_path:
                    source_type = "qq_private"
                elif "QQ群聊" in rel_path:
                    source_type = "qq_group"
                elif "微信" in rel_path or "wechat" in rel_path:
                    source_type = "wechat"
                elif "QQ" in rel_path or "qq" in rel_path:
                    source_type = "qq_private"  # 兼容旧的QQ记录目录
                else:
                    source_type = "auto"
                txt_files.append((path, source_type))
    
    return txt_files

def filter_by_date_range(input_dir, start_date_str, end_date_str, qq_id=None, target_username=None):
    """
    根据日期范围过滤聊天记录，并删除没有自己发言的文件。
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

    # 处理target_username配置
    if isinstance(target_username, dict):
        qq_username = target_username.get("qq", "")
        wechat_private_username = target_username.get("wechat_private", "")
        wechat_group_username = target_username.get("wechat_group", "")
        if not wechat_private_username and not wechat_group_username:
            wechat_private_username = target_username.get("wechat", "")
            wechat_group_username = target_username.get("wechat", "")
    else:
        qq_username = target_username or ""
        wechat_private_username = target_username or ""
        wechat_group_username = target_username or ""

    txt_files = get_all_txt_files(input_dir)
    
    for path, source_type in txt_files:
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            messages = parse_chat_content(content, source_type)
            if not messages:
                # 空文件，删除
                logger.info(f"文件为空，删除: {os.path.basename(path)}")
                os.remove(path)
                continue

            # 过滤消息
            filtered_messages = [msg for msg in messages if start_date <= msg["timestamp"] <= end_date]
            
            if not filtered_messages:
                # 日期范围内没有消息，删除
                logger.info(f"日期范围内无消息，删除: {os.path.basename(path)}")
                os.remove(path)
                continue

            # 检查是否包含自己的发言
            has_own_message = False
            for msg in filtered_messages:
                if source_type == "qq_group":
                    # QQ群聊：通过QQ号匹配
                    if qq_id and f"({qq_id})" in msg["sender"]:
                        has_own_message = True
                        break
                elif source_type == "qq_private":
                    # QQ私聊：通过用户名匹配
                    if qq_username and msg["sender"] == qq_username:
                        has_own_message = True
                        break
                elif source_type == "wechat":
                    # 微信：通过用户名匹配
                    if msg["sender"] == "我" or msg["sender"] == wechat_group_username:
                        has_own_message = True
                        break

            if not has_own_message:
                # 没有自己的发言，删除
                logger.info(f"无自己发言，删除: {os.path.basename(path)}")
                os.remove(path)
                continue

            # 需要更新文件内容（过滤日期）
            if len(filtered_messages) < len(messages):
                new_content = ""
                for msg in filtered_messages:
                    new_content += f"{msg['timestamp_str']} {msg['sender']}\n{msg['text']}\n\n"
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.info(f"已完成日期过滤: {os.path.basename(path)} (保留了 {len(filtered_messages)}/{len(messages)} 条消息)")
            
        except Exception as e:
            logger.error(f"过滤文件 {path} 失败: {e}")

def is_group_chat(filename, source_type):
    """
    判断文件是否为群聊记录。
    微信：文件名以"群聊_"开头
    QQ群聊：source_type为"qq_group"
    """
    if source_type == "wechat":
        return "群聊" in filename
    elif source_type == "qq_group":
        return True
    else:
        return False

def preprocess_group_chats(input_dir, target_username, context_lines, qq_id=None):
    """
    预处理群聊文件：保留目标用户发言及其前后的上下文。
    支持微信和QQ两种格式。
    target_username 可以是字符串（向后兼容）或字典 {"qq": "用户名", "wechat_private": "用户名", "wechat_group": "用户名"}
    qq_id: QQ号，用于在QQ群聊中识别用户（因为群名片可能不同）
    """
    if not os.path.exists(input_dir):
        return

    # 处理target_username配置
    if isinstance(target_username, dict):
        qq_username = target_username.get("qq", "")
        wechat_private_username = target_username.get("wechat_private", "")
        wechat_group_username = target_username.get("wechat_group", "")
        # 向后兼容：如果没有分别配置，则使用wechat字段
        if not wechat_private_username and not wechat_group_username:
            wechat_private_username = target_username.get("wechat", "")
            wechat_group_username = target_username.get("wechat", "")
    else:
        # 向后兼容：如果是字符串，所有来源使用相同的用户名
        qq_username = target_username
        wechat_private_username = target_username
        wechat_group_username = target_username

    txt_files = get_all_txt_files(input_dir)
    
    for path, source_type in txt_files:
        filename = os.path.basename(path)
        
        # 判断是否为群聊
        if not is_group_chat(filename, source_type):
            continue
        
        # 根据来源类型选择匹配方式
        if source_type == "qq_group":
            # QQ群聊使用QQ号匹配
            if not qq_id:
                logger.warning(f"未配置QQ号(qq_id)，跳过QQ群聊预处理: {filename}")
                continue
            use_qq_id_match = True
            match_identifier = qq_id
        elif source_type == "wechat":
            # 微信群聊使用用户名匹配
            use_qq_id_match = False
            match_identifier = wechat_group_username
        else:
            continue
            
        if not match_identifier:
            logger.warning(f"未配置目标用户标识，跳过群聊预处理: {filename}")
            continue
            
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            messages = parse_chat_content(content, source_type)
            if not messages:
                continue

            # 找出目标用户的发言索引
            if use_qq_id_match:
                # QQ群聊：通过QQ号匹配（sender格式为"群名片(QQ号)"）
                target_indices = [i for i, msg in enumerate(messages) 
                                  if f"({match_identifier})" in msg["sender"]]
            else:
                # 微信群聊：通过用户名匹配
                target_indices = [i for i, msg in enumerate(messages) 
                                  if msg["sender"] == match_identifier]
            
            if not target_indices:
                # 如果文件中没有目标用户的发言，删除文件
                logger.info(f"文件 {filename} 中未找到目标用户 ({match_identifier})，删除文件。")
                os.remove(path)
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

async def main(config_path="config.json"):
    # 1. 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    if not config:
        return

    # 为每个config创建单独的日志文件
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 添加文件日志处理器
    log_file = os.path.join(log_dir, f"{config_name}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 失败批次记录目录
    failed_dir = os.path.join(log_dir, "failed_batches")
    os.makedirs(failed_dir, exist_ok=True)
    
    logger.info(f"日志文件: {log_file}")
    logger.info(f"失败批次目录: {failed_dir}")

    api_cfg = config["api"]
    prompt_cfg = config["prompts"]
    file_cfg = config["files"]

    # 2. 预处理：日期范围过滤（同时删除无自己发言的文件）
    logger.info("开始日期范围过滤...")
    filter_by_date_range(
        file_cfg["input_dir"],
        file_cfg.get("start_date", "2000-01-01"),
        file_cfg.get("end_date", "2099-12-31"),
        qq_id=file_cfg.get("qq_id"),
        target_username=file_cfg.get("target_username")
    )

    # 3. 预处理：群聊文件
    logger.info("开始群聊预处理...")
    preprocess_group_chats(
        file_cfg["input_dir"], 
        file_cfg.get("target_username", ""), 
        file_cfg.get("context_lines", 10),
        qq_id=file_cfg.get("qq_id")
    )

    # 4. 初始化数据库（支持增量更新）
    db_path = file_cfg["output_db"]
    
    # 如果数据库已存在，备份为 原名_origin_时间戳
    if os.path.exists(db_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.splitext(db_path)[0]}_origin_{timestamp}.db"
        shutil.copy2(db_path, backup_name)
        logger.info(f"已备份原始数据库: {backup_name}")
    
    db = MemoryDB(db_path)
    
    # 显示已有数据库信息
    if os.path.exists(db_path):
        existing_nodes = db.get_all_nodes()
        existing_events_count = db.get_events_count()
        print("\n" + "="*50)
        print(f"检测到已有数据库: {db_path}")
        print(f"已有事件数: {existing_events_count}")
        print(f"已有节点数: {len(existing_nodes)}")
        print("将在此基础上继续更新。")
        print("="*50)
    else:
        logger.info(f"创建新数据库: {db_path}")
    
    # 5. 扫描并解析所有聊天文件
    input_dir = file_cfg["input_dir"]
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return

    all_blocks = []
    txt_files = get_all_txt_files(input_dir)
    
    for path, source_type in sorted(txt_files, key=lambda x: x[0]):
        filename = os.path.basename(path)
        try:
            with open(path, 'r', encoding='utf-8-sig') as f: # 处理可能的 BOM
                content = f.read()
                messages = parse_chat_content(content, source_type)
                if not messages:
                    continue
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
    print(f"配置文件: {config_path}")
    print(f"待处理文本总长度: {total_chars} 字符")
    print(f"预计消耗 Token 数 (输入): 约 {min_tokens} ~ {max_tokens}")
    print(f"分批数量: {len(batches)} 批")
    print("="*50)
    
    if not AUTO_CONFIRM:
        user_input = input("是否确定开始总结？(y/n): ").strip().lower()
        if user_input != 'y':
            print("已取消处理。")
            return
    else:
        print("AUTO_CONFIRM 已开启，自动开始总结...")

    # 8. 初始化 LLM 客户端（禁用内置重试，使用我们自己的重试逻辑）
    client = AsyncOpenAI(
        api_key=api_cfg["api_key"], 
        base_url=api_cfg["base_url"],
        timeout=180.0,  # 3分钟超时，给更多时间
        max_retries=0,  # 禁用内置重试
        default_headers={
            "User-Agent": "Cursor/0.45.0 (Windows_NT; x64) AppleWebKit/537.36"
        }
    )
    
    # 请求频率控制
    request_semaphore = asyncio.Semaphore(2)  # 同一时间最多2个请求在API端
    last_request_time = 0
    request_lock = asyncio.Lock()

    async def llm_generate(prompt, system_prompt):
        """调用LLM API，支持自动重试和反风控"""
        max_retries = file_cfg.get("max_retries", 50)
        
        for attempt in range(max_retries):
            try:
                # 频率控制：确保请求之间有足够间隔
                async with request_lock:
                    nonlocal last_request_time
                    now = asyncio.get_event_loop().time()
                    elapsed = now - last_request_time
                    # 至少间隔3-8秒（随机）
                    min_interval = 3 + random.uniform(0, 5)
                    if elapsed < min_interval:
                        wait = min_interval - elapsed
                        logger.debug(f"等待 {wait:.1f} 秒以控制请求频率...")
                        await asyncio.sleep(wait)
                    last_request_time = asyncio.get_event_loop().time()
                
                # 限制同时发送的请求数
                async with request_semaphore:
                    logger.info(f"正在调用LLM API... (prompt长度: {len(prompt)} 字符, 尝试 {attempt+1}/{max_retries})")
                    response = await client.chat.completions.create(
                        model=api_cfg["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    logger.info("LLM API调用成功")
                    return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 前5次：指数退避 2, 8, 16, 32, 64秒
                    # 之后：120-420秒随机
                    if attempt < 5:
                        wait_times = [2, 8, 16, 32, 64]
                        wait_time = wait_times[attempt]
                    else:
                        wait_time = random.uniform(0.5, 5)
                    
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"LLM API调用失败，已重试 {max_retries} 次，放弃该批次")
                    raise

    # 9. 执行总结（并行处理）
    summarizer = DailySummarizer(
        llm_generate, 
        ai_name=prompt_cfg["ai_name"], 
        base_system_prompt=prompt_cfg["system_prompt"]
    )
    
    # 并发控制：动态根据CPU情况或使用配置值
    max_concurrent_config = file_cfg.get("max_concurrent", 0)  # 0表示自动
    max_concurrent = get_dynamic_concurrency(max_concurrent_config)
    semaphore = asyncio.Semaphore(max_concurrent)
    db_lock = asyncio.Lock()  # 保护数据库写入
    
    logger.info(f"开始处理，共分为 {len(batches)} 个批次，并发数: {max_concurrent}")
    
    # 记录失败的批次
    failed_batches = []
    failed_lock = asyncio.Lock()
    
    async def process_batch(i, batch_text):
        """处理单个批次"""
        async with semaphore:
            logger.info(f"正在处理批次 {i+1}/{len(batches)} (大小: {len(batch_text.encode('utf-8'))/1024:.2f} KB)...")
            
            try:
                # 每次总结前重新获取最新节点背景
                existing_nodes = db.get_all_nodes()
                
                # 智能匹配相关节点
                max_nodes = file_cfg.get("max_nodes_context", 50)
                relevant_nodes = match_relevant_nodes(existing_nodes, batch_text, max_nodes)
                
                if relevant_nodes:
                    logger.info(f"批次 {i+1}: 从 {len(existing_nodes)} 个节点中匹配到 {len(relevant_nodes)} 个相关节点")
                    nodes_context = format_nodes_context(relevant_nodes)
                else:
                    logger.info(f"批次 {i+1}: 未匹配到相关节点")
                    nodes_context = ""

                logger.info(f"批次 {i+1}: 正在调用LLM...")
                batch_summary = await summarizer.generate_batch_summary(batch_text, nodes_context)

                if batch_summary and batch_summary.events:
                    logger.info(f"批次 {i+1} LLM返回: {len(batch_summary.events)} 个事件, {len(batch_summary.nodes)} 个节点")
                    # 使用锁保护数据库写入
                    async with db_lock:
                        db.insert_summary(batch_summary)
                    logger.info(f"批次 {i+1}: 已写入数据库")
                else:
                    logger.error(f"批次 {i+1} 总结生成失败，请检查API连接或日志。")
                    # 记录失败批次
                    async with failed_lock:
                        failed_batches.append(i)
                    # 保存失败批次内容到文件
                    failed_file = os.path.join(failed_dir, f"{config_name}_batch_{i+1}.txt")
                    with open(failed_file, 'w', encoding='utf-8') as f:
                        f.write(batch_text)
                    logger.info(f"批次 {i+1}: 失败内容已保存到 {failed_file}")
            except Exception as e:
                logger.error(f"批次 {i+1} 处理异常: {e}")
                # 记录失败批次
                async with failed_lock:
                    failed_batches.append(i)
                # 保存失败批次内容到文件
                failed_file = os.path.join(failed_dir, f"{config_name}_batch_{i+1}.txt")
                with open(failed_file, 'w', encoding='utf-8') as f:
                    f.write(batch_text)
                logger.info(f"批次 {i+1}: 失败内容已保存到 {failed_file}")
    
    # 并行执行所有批次
    tasks = [process_batch(i, batch_text) for i, batch_text in enumerate(batches)]
    await asyncio.gather(*tasks)
    
    # 记录失败批次汇总
    if failed_batches:
        failed_summary_file = os.path.join(failed_dir, f"{config_name}_failed_summary.txt")
        with open(failed_summary_file, 'w', encoding='utf-8') as f:
            f.write(f"配置文件: {config_path}\n")
            f.write(f"总批次数: {len(batches)}\n")
            f.write(f"失败批次数: {len(failed_batches)}\n")
            f.write(f"失败批次列表: {[i+1 for i in failed_batches]}\n")
            f.write(f"\n重新运行失败批次的命令:\n")
            f.write(f"python main.py --retry {config_name}\n")
        logger.warning(f"共有 {len(failed_batches)} 个批次失败，详情见 {failed_summary_file}")
    else:
        logger.info("所有批次处理成功！")

    # 所有批次完成后，进行节点优化（分批处理）
    # 优化前备份数据库
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{os.path.splitext(db_path)[0]}_{timestamp}.db"
    shutil.copy2(db_path, backup_name)
    logger.info(f"已备份数据库（优化前）: {backup_name}")
    
    logger.info("开始节点优化...")
    all_nodes = db.get_all_nodes_list()
    optimize_batch_size = file_cfg.get("optimize_batch_size", 20)
    
    # 按类型分组，同类节点一起优化
    nodes_by_type = {}
    for node in all_nodes:
        node_type = node.get('type', '未知') or '未知'
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node)
    
    optimized_count = 0
    deleted_count = 0
    
    for node_type, nodes in nodes_by_type.items():
        # 将同类型节点分批
        for batch_start in range(0, len(nodes), optimize_batch_size):
            batch_nodes = nodes[batch_start:batch_start + optimize_batch_size]
            logger.info(f"正在优化 {node_type} 类型节点: {batch_start+1}-{min(batch_start+optimize_batch_size, len(nodes))}/{len(nodes)}")
            
            result = await summarizer.optimize_nodes_batch(batch_nodes)
            
            if result:
                # 更新优化后的节点
                for opt_node in result.get('optimized_nodes', []):
                    db.update_node(
                        name=opt_node['name'],
                        type=opt_node.get('type'),
                        description=opt_node.get('description'),
                        aliases=opt_node.get('aliases'),
                        related_events=opt_node.get('related_events')
                    )
                    optimized_count += 1
                
                # 删除冗余节点
                nodes_to_delete = result.get('nodes_to_delete', [])
                if nodes_to_delete:
                    db.delete_nodes(nodes_to_delete)
                    deleted_count += len(nodes_to_delete)
            else:
                logger.warning(f"节点优化批次失败，跳过")
    
    logger.info(f"节点优化完成: 优化了 {optimized_count} 个节点，删除了 {deleted_count} 个冗余节点")
    logger.info(f"所有批次处理完成。数据库: {file_cfg['output_db']}")
    
    # 移除文件日志处理器，避免重复添加
    logger.removeHandler(file_handler)
    file_handler.close()

async def retry_failed_batches(config_path="config.json"):
    """重新运行失败的批次"""
    # 1. 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    if not config:
        return
    
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    failed_dir = os.path.join("logs", "failed_batches")
    
    # 为retry创建日志文件
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{config_name}_retry.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 查找失败批次文件
    failed_files = []
    if os.path.exists(failed_dir):
        for f in os.listdir(failed_dir):
            if f.startswith(f"{config_name}_batch_") and f.endswith(".txt"):
                failed_files.append(os.path.join(failed_dir, f))
    
    if not failed_files:
        logger.info(f"没有找到 {config_name} 的失败批次")
        logger.removeHandler(file_handler)
        file_handler.close()
        return
    
    logger.info(f"找到 {len(failed_files)} 个失败批次，开始重新运行...")
    
    # 按文件名排序（批次号顺序）
    failed_files.sort()
    
    api_cfg = config["api"]
    prompt_cfg = config["prompts"]
    file_cfg = config["files"]
    
    # 初始化数据库
    db_path = file_cfg["output_db"]
    
    # 备份数据库
    if os.path.exists(db_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.splitext(db_path)[0]}_retry_{timestamp}.db"
        shutil.copy2(db_path, backup_name)
        logger.info(f"已备份数据库: {backup_name}")
    
    db = MemoryDB(db_path)
    
    # 初始化LLM客户端
    client = AsyncOpenAI(
        api_key=api_cfg["api_key"], 
        base_url=api_cfg["base_url"],
        timeout=180.0,
        max_retries=0,
        default_headers={
            "User-Agent": "Cursor/0.45.0 (Windows_NT; x64) AppleWebKit/537.36"
        }
    )
    
    # 请求频率控制
    request_semaphore = asyncio.Semaphore(2)
    last_request_time = 0
    request_lock = asyncio.Lock()
    
    async def llm_generate(prompt, system_prompt):
        """调用LLM API"""
        max_retries = file_cfg.get("max_retries", 50)
        
        for attempt in range(max_retries):
            try:
                async with request_lock:
                    nonlocal last_request_time
                    now = asyncio.get_event_loop().time()
                    elapsed = now - last_request_time
                    min_interval = 3 + random.uniform(0, 5)
                    if elapsed < min_interval:
                        wait = min_interval - elapsed
                        await asyncio.sleep(wait)
                    last_request_time = asyncio.get_event_loop().time()
                
                async with request_semaphore:
                    logger.info(f"正在调用LLM API... (prompt长度: {len(prompt)} 字符, 尝试 {attempt+1}/{max_retries})")
                    response = await client.chat.completions.create(
                        model=api_cfg["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    logger.info("LLM API调用成功")
                    return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    if attempt < 5:
                        wait_times = [2, 8, 16, 32, 64]
                        wait_time = wait_times[attempt]
                    else:
                        wait_time = random.uniform(0.5, 5)
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"LLM API调用失败，已重试 {max_retries} 次")
                    raise
    
    summarizer = DailySummarizer(
        llm_generate, 
        ai_name=prompt_cfg["ai_name"], 
        base_system_prompt=prompt_cfg["system_prompt"]
    )
    
    # 并发控制
    max_concurrent = 2  # retry时使用较低并发
    semaphore = asyncio.Semaphore(max_concurrent)
    db_lock = asyncio.Lock()
    
    # 读取批次内容
    batches = []
    for batch_file in failed_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batches.append(f.read())
    
    logger.info(f"开始重新运行 {len(batches)} 个失败批次")
    
    # 记录失败的批次
    failed_batches = []
    failed_lock = asyncio.Lock()
    
    async def process_batch(i, batch_text, batch_file):
        """处理单个批次"""
        async with semaphore:
            logger.info(f"正在重新处理批次 {i+1}/{len(batches)} (来自 {os.path.basename(batch_file)})...")
            
            try:
                existing_nodes = db.get_all_nodes()
                max_nodes = file_cfg.get("max_nodes_context", 50)
                relevant_nodes = match_relevant_nodes(existing_nodes, batch_text, max_nodes)
                
                if relevant_nodes:
                    nodes_context = format_nodes_context(relevant_nodes)
                else:
                    nodes_context = ""

                batch_summary = await summarizer.generate_batch_summary(batch_text, nodes_context)

                if batch_summary and batch_summary.events:
                    logger.info(f"批次 {i+1} LLM返回: {len(batch_summary.events)} 个事件, {len(batch_summary.nodes)} 个节点")
                    async with db_lock:
                        db.insert_summary(batch_summary)
                    logger.info(f"批次 {i+1}: 已写入数据库")
                    # 成功后删除失败文件
                    os.remove(batch_file)
                    logger.info(f"批次 {i+1}: 已删除失败文件 {os.path.basename(batch_file)}")
                else:
                    logger.error(f"批次 {i+1} 重新处理仍然失败")
                    async with failed_lock:
                        failed_batches.append(i)
            except Exception as e:
                logger.error(f"批次 {i+1} 处理异常: {e}")
                async with failed_lock:
                    failed_batches.append(i)
    
    # 并行执行所有批次
    tasks = [process_batch(i, batch_text, failed_files[i]) for i, batch_text in enumerate(batches)]
    await asyncio.gather(*tasks)
    
    # 记录结果
    if failed_batches:
        logger.warning(f"仍有 {len(failed_batches)} 个批次失败")
    else:
        logger.info("所有失败批次重新处理成功！")
    
    # 移除日志处理器
    logger.removeHandler(file_handler)
    file_handler.close()

if __name__ == "__main__":
    # 命令行参数用法:
    # python main.py                    # 使用默认 config.json
    # python main.py config1            # 使用 config1.json
    # python main.py config1 config2    # 依次使用 config1.json 和 config2.json
    # python main.py --retry config1    # 重新运行 config1 的失败批次
    
    if len(sys.argv) > 1:
        # 检查是否有 --retry 参数
        if sys.argv[1] == "--retry":
            # 重新运行失败批次
            if len(sys.argv) > 2:
                config_names = sys.argv[2:]
                for config_name in config_names:
                    if not config_name.endswith('.json'):
                        config_file = f"{config_name}.json"
                    else:
                        config_file = config_name
                    
                    if not os.path.exists(config_file):
                        logger.error(f"配置文件不存在: {config_file}")
                        continue
                    
                    print(f"\n{'='*60}")
                    print(f"重新运行失败批次: {config_file}")
                    print(f"{'='*60}\n")
                    
                    asyncio.run(retry_failed_batches(config_file))
            else:
                # 默认使用 config.json
                asyncio.run(retry_failed_batches())
        else:
            # 正常运行
            config_names = sys.argv[1:]
            for config_name in config_names:
                if not config_name.endswith('.json'):
                    config_file = f"{config_name}.json"
                else:
                    config_file = config_name
                
                if not os.path.exists(config_file):
                    logger.error(f"配置文件不存在: {config_file}")
                    continue
                
                print(f"\n{'='*60}")
                print(f"开始处理配置文件: {config_file}")
                print(f"{'='*60}\n")
                
                asyncio.run(main(config_file))
    else:
        # 默认使用 config.json
        asyncio.run(main())
