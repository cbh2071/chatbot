# protein_utils.py
import httpx  # 导入httpx库，用于发送异步HTTP请求
import re     # 导入re库，用于正则表达式操作
import logging # 导入logging库，用于记录程序运行信息
import asyncio # 导入asyncio库，用于支持异步操作
import json    # 导入json库，用于解析JSON数据

# 配置基本的日志记录器
# 日志级别设置为INFO，意味着INFO及以上级别（WARNING, ERROR, CRITICAL）的日志都会被记录
# 日志格式包含时间、记录器名称、日志级别和日志消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 获取名为__name__（即当前模块名'protein_utils'）的日志记录器实例
logger = logging.getLogger(__name__)

# 定义用于匹配标准氨基酸（包括'-'代表的gap）的正则表达式，不区分大小写
# ^ 表示字符串开头，$ 表示字符串结尾，+ 表示一个或多个字符
VALID_AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY-]+$", re.IGNORECASE)

# 定义用于匹配典型UniProtKB登录号（例如 P12345, Q9Y6Q9, A0A024R1R8）的正则表达式
# 也包括亚型标识符，如 P12345-1
# [A-Z0-9]{6,10} 匹配6到10个大写字母或数字
# (?:-\d+)? 匹配可选的连字符和其后的一个或多个数字（非捕获组）
UNIPROT_ID_PATTERN = re.compile(r"^[A-Z0-9]{6,10}(?:-\d+)?$", re.IGNORECASE)

# 定义一个更简单的检查，用于匹配UniProt入口名称，格式通常是 '名称_物种' (例如 'INS_HUMAN')
# [A-Z0-9]+ 匹配一个或多个大写字母或数字
# _ 匹配下划线
UNIPROT_NAME_PATTERN = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$", re.IGNORECASE)

def validate_sequence(sequence: str) -> bool:
    """
    验证输入字符串是否是一个合法的蛋白质序列。
    :param sequence: 输入的待验证字符串。
    :return: 如果字符串符合蛋白质序列格式（仅包含标准氨基酸字符和'-'），返回True，否则返回False。
    """
    if not sequence:
        logger.warning("序列验证失败：输入序列为空。")
        return False
    # 使用正则表达式匹配整个序列
    if VALID_AA_PATTERN.match(sequence):
        # 可选：在这里添加序列长度检查，如果模型有特定要求
        # 例如，如果模型只接受长度在10到10000之间的序列：
        # if not (10 <= len(sequence) <= 10000):
        #     logger.warning(f"序列验证警告：序列长度 {len(sequence)} 不在常规范围内(10-10000)。")
        # 如果长度检查也通过（或没有长度检查），则认为序列有效
        return True
    else:
        # 如果匹配失败，记录日志并找出无效字符
        # 只记录序列开头部分以避免日志过大
        # 使用集合找出所有不在有效模式中的字符
        invalid_chars = set(c.upper() for c in sequence if not VALID_AA_PATTERN.match(c))
        logger.warning(f"序列验证失败：序列包含无效字符: {invalid_chars}。序列开头: {sequence[:30]}...")
        return False

def is_potential_uniprot_id(text: str) -> bool:
    """
    检查给定的文本是否看起来像一个UniProt登录号或入口名称。
    :param text: 需要检查的文本字符串。
    :return: 如果文本符合UniProt ID或Name的模式，返回True，否则返回False。
    """
    # 使用前面定义的两个正则表达式进行匹配
    return bool(UNIPROT_ID_PATTERN.match(text) or UNIPROT_NAME_PATTERN.match(text))

async def fetch_protein_data(uniprot_id_or_name: str) -> dict | None:
    """
    使用UniProt登录号或入口名称从UniProt API异步获取蛋白质序列和来源物种信息。
    :param uniprot_id_or_name: UniProt登录号 (如 "P00533") 或入口名称 (如 "INS_HUMAN")。
    :return: 如果成功获取并验证数据，返回包含 'sequence', 'organism', 'id' 的字典；
             如果未找到条目、获取失败或数据无效，返回 None。
    """
    identifier = uniprot_id_or_name.strip() # 去除首尾空格，保留原始大小写以备名称查询
    # 使用 UniProt 最新的 REST API 搜索端点
    # 构建查询URL，同时搜索登录号(accession)和ID/名称(id)字段
    # fields参数指定需要返回的字段：登录号、ID、物种名、序列
    # format=json指定返回JSON格式，size=1表示只取第一个匹配结果
    url = f"https://rest.uniprot.org/uniprotkb/search?query=accession:{identifier}%20OR%20id:{identifier.upper()}&fields=accession,id,organism_name,sequence&format=json&size=1"

    # 创建一个异步HTTP客户端实例，设置超时时间为20秒，并允许自动处理重定向
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        try:
            logger.info(f"开始从UniProt获取数据: {identifier}")
            # 发送GET请求
            response = await client.get(url)
            # 检查响应状态码，如果不是2xx成功状态，则抛出HTTPStatusError异常
            response.raise_for_status()
            # 解析JSON响应体
            data = response.json()

            # 从响应数据中获取结果列表
            results = data.get("results")
            if not results:
                # 如果结果列表为空，表示未找到对应的UniProt条目
                logger.warning(f"未找到 UniProt 条目: {identifier}")
                return None

            # 取结果列表中的第一个条目
            entry = results[0]
            # 从条目中提取序列信息，注意嵌套结构和可能的缺失
            sequence = entry.get("sequence", {}).get("value")
            # 从条目中提取物种信息，提供默认值
            organism = entry.get("organism", {}).get("scientificName", "Unknown Organism")
            # 获取主要的登录号，作为规范标识符返回
            accession = entry.get("primaryAccession", identifier)

            if not sequence:
                # 如果序列字段缺失或为空
                logger.error(f"在 UniProt 条目中未找到序列: {identifier} (登录号: {accession})")
                return None

            # 对从API获取到的序列进行验证，确保其符合标准格式
            if not validate_sequence(sequence):
                 logger.error(f"获取到的序列 {identifier} (登录号: {accession}) 未通过验证。序列开头: {sequence[:30]}...")
                 # 决定是返回无效序列还是None，这里选择严格模式，不返回无效序列
                 return None

            # 成功获取并验证数据后，记录日志
            logger.info(f"成功获取数据: {identifier} (登录号: {accession}, 物种: {organism}, 序列长度: {len(sequence)})")
            # 返回包含序列、物种和主要登录号的字典
            return {
                "sequence": sequence,
                "organism": organism,
                "id": accession # 返回实际获取到的主要登录号
            }
        # 捕获并处理HTTP状态错误（如404 Not Found, 500 Internal Server Error）
        except httpx.HTTPStatusError as e:
            logger.error(f"获取 {identifier} 时发生HTTP错误: 状态码 {e.response.status_code} - {e.response.text[:200]}")
            return None
        # 捕获并处理网络请求相关的错误（如DNS解析失败、连接超时）
        except httpx.RequestError as e:
            logger.error(f"获取 {identifier} 时发生网络错误: {e}")
            return None
        # 捕获并处理JSON解析错误
        except json.JSONDecodeError as e:
             logger.error(f"解析 {identifier} 的JSON响应失败: {e}")
             return None
        # 捕获其他所有预料之外的异常
        except Exception as e:
            # 使用 logger.exception 会同时记录错误信息和堆栈跟踪，便于调试
            logger.exception(f"处理 {identifier} 的UniProt数据时发生意外错误: {e}")
            return None

# 这个模块如果直接运行（`python protein_utils.py`），下面的代码会执行，用于测试
if __name__ == "__main__":
    async def run_test():
        print("测试 fetch_protein_data:")
        result1 = await fetch_protein_data("P00533") # 测试UniProt ID
        print(f"P00533 结果: {result1}")
        result2 = await fetch_protein_data("INS_HUMAN") # 测试UniProt Entry Name
        print(f"INS_HUMAN 结果: {result2}")
        result3 = await fetch_protein_data("nonexistent") # 测试无效ID
        print(f"nonexistent 结果: {result3}")
        print("\n测试 validate_sequence:")
        print(f"有效序列 'ACDEF': {validate_sequence('ACDEF')}")
        print(f"无效序列 'ACDEFZ': {validate_sequence('ACDEFZ')}")
        print(f"空序列 '': {validate_sequence('')}")
    asyncio.run(run_test())