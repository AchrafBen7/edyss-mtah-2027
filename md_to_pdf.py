"""Render a Markdown file to a clean PDF (reportlab Platypus). Handles headers, paragraphs, pipe
tables, fenced code, bullets and numbered lists (wrapped lines included), inline **bold** / *italic* /
`code` / [links](https://...). Unicode is mapped to ASCII to avoid missing glyphs.
Needs reportlab: pip install -r requirements-extra.txt

    python md_to_pdf.py WRITEUP.md edyss.pdf
"""
import re
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
USABLE = 174 * mm

UNI = {"—": "-", "–": "-", "→": "->", "×": "x", "≈": "~", "·": "-",
       "²": "^2", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
       "€": "EUR", "•": "-", "≠": "!=", "±": "+/-", "≥": ">=", "≤": "<=",
       "−": "-", "Δ": "delta "}


def clean(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return "".join(c if ord(c) < 256 else "?" for c in s)


def inline(s):
    s = clean(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # code spans first, with * and _ escaped so the emphasis rules below cannot touch them
    s = re.sub(r"`([^`]+?)`", lambda m: '<font face="Courier" size="8.3">{}</font>'.format(
        m.group(1).replace("*", "&#42;").replace("_", "&#95;")), s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<link href="\2" color="#26527f">\1</link>', s)
    return s


ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Title'], fontSize=17, spaceAfter=4, alignment=0,
                    textColor=colors.HexColor('#16314f'))
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=12.5, spaceBefore=11, spaceAfter=4,
                    textColor=colors.HexColor('#26527f'))
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontSize=10.5, spaceBefore=7, spaceAfter=2)
BODY = ParagraphStyle('BODY', parent=ss['BodyText'], fontSize=9.3, leading=12.6, spaceAfter=5, alignment=4)
CODE = ParagraphStyle('CODE', parent=ss['Code'], fontSize=7.8, leading=9.6, textColor=colors.HexColor('#333333'))
CELL = ParagraphStyle('CELL', parent=ss['BodyText'], fontSize=8, leading=10.5)
CELLH = ParagraphStyle('CELLH', parent=CELL, textColor=colors.white, fontName='Helvetica-Bold')

lines = text.split("\n")
story = []
i, n = 0, len(lines)


LIST_ITEM = re.compile(r"\s*(-|\d+\.) (.*)")


def mk_bullets(items, marks):
    data = [[Paragraph(mk, CELL), Paragraph(inline(it), BODY)] for mk, it in zip(marks, items)]
    t = Table(data, colWidths=[6 * mm, USABLE - 6 * mm], hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0),
                           ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 1)]))
    return t


while i < n:
    ln = lines[i]
    if ln.strip() == "":
        i += 1; continue
    if ln.startswith("```"):
        j = i + 1; buf = []
        while j < n and not lines[j].startswith("```"):
            buf.append(clean(lines[j])); j += 1
        story.append(Preformatted("\n".join(buf), CODE, maxLineLength=104))
        story.append(Spacer(1, 5)); i = j + 1; continue
    if ln.startswith("# "):
        story.append(Paragraph(inline(ln[2:]), H1)); i += 1; continue
    if ln.startswith("## "):
        story.append(Paragraph(inline(ln[3:]), H2)); i += 1; continue
    if ln.startswith("### "):
        story.append(Paragraph(inline(ln[4:]), H3)); i += 1; continue
    if ln.lstrip().startswith("|") and i + 1 < n and \
            set(lines[i + 1].replace("|", "").replace("-", "").replace(":", "").strip()) <= set(" "):
        rows = []
        while i < n and lines[i].lstrip().startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
        header, bodyr = rows[0], rows[2:]
        nc = len(header)
        cw = [0.46 * USABLE, 0.54 * USABLE] if nc == 2 else [USABLE / nc] * nc
        data = [[Paragraph(inline(c), CELLH) for c in header]] + \
               [[Paragraph(inline(c), CELL) for c in r] for r in bodyr]
        t = Table(data, colWidths=cw, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#26527f')),
                               ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
                               ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6fb')]),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                               ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
        story.append(t); story.append(Spacer(1, 6)); continue
    if LIST_ITEM.match(ln):
        items, marks = [], []
        while i < n and LIST_ITEM.match(lines[i]):
            m = LIST_ITEM.match(lines[i])
            marks.append(m.group(1)); items.append(m.group(2)); i += 1
            # wrapped continuation lines (indented, non-empty, not a new item) belong to this item
            while i < n and lines[i][:1] in (" ", "\t") and lines[i].strip() and not LIST_ITEM.match(lines[i]):
                items[-1] += " " + lines[i].strip(); i += 1
        story.append(mk_bullets(items, marks)); story.append(Spacer(1, 4)); continue
    buf = [ln]; i += 1
    while i < n and lines[i].strip() != "" and not LIST_ITEM.match(lines[i]) and \
            not lines[i].lstrip().startswith(("#", "```", "|", ">")):
        buf.append(lines[i]); i += 1
    story.append(Paragraph(inline(" ".join(buf)), BODY))

SimpleDocTemplate(out, pagesize=A4, topMargin=16 * mm, bottomMargin=16 * mm,
                  leftMargin=18 * mm, rightMargin=18 * mm,
                  title="edyss - possession classification").build(story)
print("ecrit", out)
