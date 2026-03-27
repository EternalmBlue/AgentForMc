from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ragformc.domain.models import RetrievedDoc


REWRITE_SYSTEM_PROMPT = """你是一个检索问句改写器。
你的任务是把用户当前问题改写成适合检索的独立问题。
规则：
1. 只输出改写后的单条问题，不要输出解释。
2. 如果当前问题已经足够独立，就保持原意并做轻微规范化。
3. 如果问题里出现“它”“这个插件”“上面那个”等代词，要结合历史对话补全到具体插件或主题。
4. 不要编造历史中没有出现过的插件名、版本号、功能或依赖。
5. 尽量保留 Minecraft 插件名、中英文别名、数据库名、版本号、核心类名等检索关键词。"""


ANSWER_SYSTEM_PROMPT = """你是一个 Minecraft 插件资料问答助手。
你需要同时参考检索证据和你自己的通用知识来回答，但必须区分二者。
规则：
1. 优先依据“检索证据”给出答案。
2. 如果检索证据足够，就直接回答，不要额外发挥。
3. 如果检索证据不足，但你基于自身知识判断大概率正确，可以补充回答。
4. 只要使用了当前检索证据之外的知识，必须明确标注“以下内容是基于模型自身知识的补充，未在当前检索证据中直接验证”。
5. 不要把模型自身知识伪装成检索证据。
6. 如果你也不确定，就明确说不确定，不要编造。
7. 使用简洁中文。"""


def format_history(history: list[BaseMessage]) -> str:
    if not history:
        return "无历史对话。"

    turns: list[tuple[str, str]] = []
    pending_question: str | None = None

    for message in history:
        content = str(message.content).strip()

        if isinstance(message, HumanMessage):
            if pending_question is not None:
                turns.append((pending_question, "（待回答）"))
            pending_question = content
            continue

        if isinstance(message, AIMessage):
            if pending_question is None:
                turns.append(("（缺失用户问题）", content))
            else:
                turns.append((pending_question, content))
                pending_question = None

    if pending_question is not None:
        turns.append((pending_question, "（待回答）"))

    if not turns:
        return "无可显示的历史对话。"

    parts: list[str] = []
    for index, (question, answer) in enumerate(turns, start=1):
        parts.append(
            f"第{index}轮用户问题：{question}\n"
            f"第{index}轮助手回答：{answer}"
        )
    return "\n\n".join(parts)


def format_docs_for_prompt(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "无检索结果。"

    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        parts.append(
            f"[证据 {index}]\n"
            f"中文名：{doc.plugin_chinese_name}\n"
            f"英文名：{doc.plugin_english_name}\n"
            f"匹配方式：{doc.match_reason}\n"
            f"距离分数：{doc.distance:.6f}\n"
            f"内容：\n{doc.content}"
        )
    return "\n\n".join(parts)
