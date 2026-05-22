"""LangChain 消息 content 归一化为字符串（多模态时 content 可能为 list）。"""


def message_content_to_str(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)
