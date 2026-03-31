from __future__ import annotations

from agent_for_mc.domain.models import PluginConfigDoc


def format_plugin_config_docs(
    docs: list[PluginConfigDoc],
    *,
    preview_chars: int,
) -> str:
    if not docs:
        return "No matching plugin config files were found."

    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        preview = " ".join(doc.content.split())
        if preview_chars > 0:
            preview = preview[:preview_chars]
        parts.append(
            f"[{index}] id={doc.id} "
            f"plugin={doc.plugin_chinese_name} / {doc.plugin_english_name}\n"
            f"path={doc.file_path}\n"
            f"reason={doc.match_reason}\n"
            f"distance={doc.distance:.6f}\n"
            f"preview={preview}"
        )
    return "\n\n".join(parts)


def serialize_docs(docs: list[PluginConfigDoc]) -> list[dict[str, str | float | int]]:
    return [
        {
            "id": doc.id,
            "plugin_chinese_name": doc.plugin_chinese_name,
            "plugin_english_name": doc.plugin_english_name,
            "file_path": doc.file_path,
            "content": doc.content,
            "distance": doc.distance,
            "match_reason": doc.match_reason,
        }
        for doc in docs
    ]
