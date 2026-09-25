#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
站点构建脚本：把根目录文件"覆盖"进 docs/（overlay），使正文中的 ../../../ 相对链接
在 MkDocs 站点中原样生效，然后生成 nav 并调用 mkdocs build。

用法：
    python tools/build_site.py            # 构建到 site/
    python tools/build_site.py --deploy   # 构建并 gh-deploy（本地手动发布用）

CI 用法见 .github/workflows/deploy-pages.yml（构建产物推到 gh-pages 分支）。
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SITE = ROOT / "site"

# 覆盖进 docs/ 的根目录文件（正文里 ../../../X.md 的目标）
OVERLAY_FILES = [
    "README.md", "ROADMAP.md", "CONTRIBUTING.md", "CHANGELOG.md",
    "GLOSSARY.md", "CODE_OF_CONDUCT.md", "LICENSE-CONTENT", "CITATION.cff",
]
OVERLAY_DIRS = ["playground", "assets", "papers", "notebooks"]

# 与仓库阅读顺序一致的 nav（overlay 后的路径）
PIECES = ["1-起步准备", "2-经典机器学习", "3-深度学习", "4-专业方向", "5-前沿与融合"]


def overlay():
    for f in OVERLAY_FILES:
        src, dst = ROOT / f, DOCS / f
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    for d in OVERLAY_DIRS:
        src, dst = ROOT / d, DOCS / d
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)


def clean_overlay():
    """从 docs/ 删除 overlay 产物（保持工作区干净，这些文件已提交过根目录版本）"""
    for f in OVERLAY_FILES:
        p = DOCS / f
        if p.exists():
            p.unlink()
    for d in OVERLAY_DIRS:
        p = DOCS / d
        if p.is_dir():
            shutil.rmtree(p)


def page_entry(md: Path) -> tuple:
    title = first_heading(md)
    return title, str(md.relative_to(DOCS)).replace("\\", "/")


def first_heading(md: Path) -> str:
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t.replace("：", " · ")
    return md.stem


def build_nav() -> list:
    nav = [("首页", "README.md"), ("🌳 知识树总导航", "知识树.md")]
    for piece in PIECES:
        pd = DOCS / piece
        if not pd.is_dir():
            continue
        items = [("篇导览", f"{piece}/README.md")]
        for ch in sorted(p for p in pd.iterdir() if p.is_dir()):
            pages = sorted(ch.glob("*.md"))
            entries = [page_entry(p) for p in pages]
            items.append((first_heading(ch / "README.md").replace(" · ", " · "),
                          {ch.name: entries}))
        nav.append((first_heading(pd / "README.md"), {piece: items}))
    nav += [
        ("📓 实战 Notebook", "notebooks/README.md"),
        ("📜 论文精读专栏", "papers/README.md"),
        ("📖 术语表", "GLOSSARY.md"),
        ("🗺️ 阅读路线", "ROADMAP.md"),
        ("🤝 贡献指南", "CONTRIBUTING.md"),
        ("📝 更新日志", "CHANGELOG.md"),
        ("行为准则", "CODE_OF_CONDUCT.md"),
    ]
    # 转成 mkdocs 需要的嵌套结构
    def to_nav(lst):
        out = []
        for title, val in lst:
            if isinstance(val, dict):
                (k, v), = val.items()
                out.append({title: to_nav(v)})
            elif isinstance(val, list):
                out.append({title: to_nav(val)})
            else:
                out.append({title: val})
        return out
    return to_nav(nav)


def main():
    overlay()
    nav = build_nav()
    yml = ROOT / "mkdocs.yml"
    text = yml.read_text(encoding="utf-8")
    import re
    text = re.sub(r"nav: \[.*\]", "nav: " + json.dumps(nav, ensure_ascii=False), text, flags=re.S)
    tmp = ROOT / "mkdocs.autonav.yml"
    tmp.write_text(text, encoding="utf-8")

    # 站点附加文件
    (DOCS / "js").mkdir(exist_ok=True)
    (DOCS / "css").mkdir(exist_ok=True)
    (DOCS / "js" / "extra.js").write_text(
        "mermaid.initialize({startOnLoad:true, theme:'dark', "
        "themeVariables:{primaryColor:'#1f6feb', primaryTextColor:'#e6edf3', "
        "lineColor:'#8b949e', fontFamily:'Microsoft YaHei, sans-serif'}});\n",
        encoding="utf-8",
    )
    (DOCS / "css" / "extra.css").write_text(
        ".md-typeset {font-family:'Microsoft YaHei','PingFang SC',sans-serif;}\n",
        encoding="utf-8",
    )
    try:
        cmd = [sys.executable, "-m", "mkdocs", "build", "-f", str(tmp), "-d", str(SITE)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout[-3000:] if r.stdout else "", end="")
        if r.returncode != 0:
            print(r.stderr[-3000:])
            return 1
        print("BUILD OK ->", SITE)
        # 兜底：正文面包屑里的 ../../../README.md 在 use_directory_urls 下未被重写，
        # 生成一个跳转文件，让这些链接落到站点首页
        (SITE / "README.md").write_text(
            '<!DOCTYPE html><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="0;url=./">'
            '<a href="./">进入首页</a>', encoding="utf-8")
        return 0
    finally:
        tmp.unlink(missing_ok=True)
        clean_overlay()
        shutil.rmtree(DOCS / "js", ignore_errors=True)
        shutil.rmtree(DOCS / "css", ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
