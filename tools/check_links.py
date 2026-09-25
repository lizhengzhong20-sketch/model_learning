#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
链接校验与自动修复工具
用法：
    python tools/check_links.py           # 仅报告死链
    python tools/check_links.py --fix     # 报告并自动修复（调整 ../ 层级后能命中存在文件的死链）

规则：仅处理相对链接（跳过 http/https/mailto/纯锚点）。修复策略：尝试增减前导 ../
的层级变体，命中存在的文件即替换；无法自动修复的链接保持原样并列入报告。
"""
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
EXTERNAL = ("http://", "https://", "mailto:")


def variants(path: str):
    """生成前导 ../ 层级增减后的候选路径（含原路径）。"""
    parts = path.split("/")
    yield path
    lead = 0
    while lead < len(parts) and parts[lead] == "..":
        lead += 1
    rest = parts[lead:]
    for extra in (1, 2, -1, -2, -lead if lead else 99):  # 最后一个：完全去掉 ../（从仓库根解析）
        k = lead + extra
        if k < 0:
            continue
        yield "/".join([".."] * k + rest)


def main(fix: bool, ci: bool = False):
    total = broken = fixed = 0
    report = []
    for md in sorted(ROOT.rglob("*.md")):
        if any(seg in md.parts for seg in (".git", "node_modules")):
            continue
        # 自动修复仅作用于 docs/ 下的内容文件；
        # 根目录与 templates/ 中的链接多为"规范示例"（展示给页面层用的正确写法），只报告不改写
        in_scope = md.relative_to(ROOT).parts[0] == "docs"
        # --ci 模式：只把 docs/ 的死链计入退出码（模板占位符与规范示例不算失败）
        if ci and not in_scope:
            continue
        text = md.read_text(encoding="utf-8")
        orig = text
        for m in LINK_RE.finditer(text):
            raw = m.group(1)
            if raw.startswith(EXTERNAL) or raw.startswith("#"):
                continue
            path = urllib.parse.unquote(raw.split("#")[0])
            anchor = raw.split("#", 1)[1] if "#" in raw else ""
            if not path:
                continue
            total += 1
            if (md.parent / path).exists():
                continue
            # 尝试修复
            if fix and in_scope:
                for v in variants(path):
                    if v != path and (md.parent / v).exists():
                        nv = v + ("#" + anchor if anchor else "")
                        text = text.replace(f"]({raw})", f"]({nv})")
                        fixed += 1
                        report.append(f"  FIXED  {md.relative_to(ROOT)} -> {raw}  =>  {nv}")
                        break
                else:
                    broken += 1
                    report.append(f"  BROKEN {md.relative_to(ROOT)} -> {raw}")
            else:
                broken += 1
                report.append(f"  {'BROKEN' if in_scope else 'NOTE  '} {md.relative_to(ROOT)} -> {raw}")
        if fix and text != orig:
            md.write_text(text, encoding="utf-8")
    print(f"检查相对链接 {total} 个；死链 {broken} 个；自动修复 {fixed} 个。")
    for line in report[:200]:
        print(line)
    if len(report) > 200:
        print(f"  ……其余 {len(report) - 200} 条略")
    return 0 if broken == 0 else 1


if __name__ == "__main__":
    sys.exit(main(fix="--fix" in sys.argv, ci="--ci" in sys.argv))
