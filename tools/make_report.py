"""Render the project report to PDF.

    python tools/make_report.py   ->   build/hedge-fund-phase0-report.pdf

Every figure here is measured by this repository's own harness and appears in
the README; nothing is restated from memory.

The diagram is a browser capture, not a re-rendered SVG. archify styles the
SVG through CSS classes, and svglib honours only presentation attributes, so
rendering the SVG directly produced a page of solid black rectangles. The
capture is checked in at docs/decision-logic.png; to refresh it after editing
the diagram:

    archify visual-check .archify/<dir>/decision-logic.html --out-dir build/capture

then crop the viewer chrome off the resulting *.light.png into that path.
"""

from __future__ import annotations

import pathlib
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
INK = colors.HexColor("#1d1c1a")
MUTED = colors.HexColor("#6b665e")
LINE = colors.HexColor("#ddd9d1")

styles = getSampleStyleSheet()


def style(name: str, **kw) -> ParagraphStyle:
    base = dict(
        name=name, fontName="Helvetica", fontSize=9.5, leading=14,
        textColor=INK, alignment=TA_LEFT,
    )
    base.update(kw)
    return ParagraphStyle(**base)


TITLE = style("t", fontName="Times-Roman", fontSize=23, leading=27, spaceAfter=2)
SUB = style("s", fontSize=10, leading=15, textColor=MUTED, spaceAfter=14)
H2 = style("h2", fontName="Helvetica-Bold", fontSize=13, leading=17,
           spaceBefore=16, spaceAfter=5)
H3 = style("h3", fontName="Helvetica-Bold", fontSize=10, leading=14,
           spaceBefore=10, spaceAfter=3)
BODY = style("b", spaceAfter=7)
NOTE = style("n", fontSize=8.5, leading=12.5, textColor=MUTED, spaceAfter=7)
CAPTION = style("c", fontSize=8, leading=11, textColor=MUTED, spaceBefore=4)


CELL = style("cell", fontSize=8.6, leading=12)
CELL_HEAD = style("cellhead", fontName="Helvetica-Bold", fontSize=7.6,
                  leading=10, textColor=MUTED)


def wrap(rows, widths, wrap_from_col=0):
    """Turn long cells into Paragraphs so reportlab wraps instead of overflowing."""
    out = []
    for r, row in enumerate(rows):
        cells = []
        for c, cell in enumerate(row):
            text = "" if cell is None else str(cell)
            long_enough = len(text) > 28 and c >= wrap_from_col
            cells.append(Paragraph(text, CELL_HEAD if r == 0 else CELL)
                         if long_enough else text)
        out.append(cells)
    return out


def table(rows, widths, highlight_last_col=False, header=True,
          text_cols=()) -> Table:
    """Numbers right, prose left.

    `text_cols` names the columns holding prose. Without it the header sits
    right-aligned over a left-aligned Paragraph, because a wrapped cell
    ignores the table's ALIGN while a short header string obeys it.
    """
    t = Table(wrap(rows, widths), colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]
    for c in text_cols:
        cmds.append(("ALIGN", (c, 0), (c, -1), "LEFT"))
    if header:
        cmds += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("FONTSIZE", (0, 0), (-1, 0), 7.6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
        ]
    if highlight_last_col:
        cmds.append(("FONTNAME", (-1, 1), (-1, -1), "Helvetica-Bold"))
    t.setStyle(TableStyle(cmds))
    return t


def diagram(max_width: float):
    """The diagram as rendered by a real browser, not re-drawn from the SVG.

    See the module docstring for why, and for how to refresh the capture.
    """
    png = ROOT / "docs" / "decision-logic.png"
    if not png.exists():
        return None
    cropped = _crop_to_drawing(png)
    from PIL import Image as PILImage

    with PILImage.open(cropped) as im:
        ratio = im.height / im.width
    return RLImage(str(cropped), width=max_width, height=max_width * ratio)


def _crop_to_drawing(png: pathlib.Path) -> pathlib.Path:
    """Trim the capture down to the drawing itself.

    The capture carries a near-white card and a faint dot grid that a plain
    alpha/white trim keeps, which left the diagram floating in half a page of
    margin. 225 on an 8-bit grey sits below the grid and above every border
    the drawing actually uses. Writes a sibling file so the capture is kept.
    """
    import numpy as np
    from PIL import Image as PILImage

    out = BUILD / (png.stem + "-trimmed.png")
    with PILImage.open(png) as im:
        a = np.asarray(im.convert("L"))
        mask = a < 225
        if not mask.any():
            return png
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        pad = 12
        box = (max(int(cols[0]) - pad, 0), max(int(rows[0]) - pad, 0),
               min(int(cols[-1]) + 1 + pad, im.width),
               min(int(rows[-1]) + 1 + pad, im.height))
        im.convert("RGB").crop(box).save(out)
    return out


def footer(canvas, doc, offset: int = 0):
    """Offset exists because the landscape page is a second document.

    The report is three concatenated documents, each of which would number
    itself from 1. Every piece is told how many pages precede it.
    """
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    page_width = canvas._pagesize[0]
    canvas.drawString(20 * mm, 12 * mm, "hedge_fund — Phase 0 report")
    canvas.drawRightString(page_width - 20 * mm, 12 * mm, f"{doc.page + offset}")
    canvas.restoreState()


def build(part: str = "a", page_offset: int = 0) -> pathlib.Path:
    """One half of the portrait body.

    Part "a" is everything up to the price-factor verdict; part "b" is
    everything after. The landscape plate is concatenated between them, so
    no paragraph can be split across it — which is what happened when the
    plate was spliced into a single continuous body at a fixed page number.
    """
    out = BUILD / f"_body_{part}.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="hedge_fund — Phase 0 report",
        author="Chaut007", subject="Systematic equity strategy: build and findings",
    )
    width = doc.width
    s = []

    s += [
        Paragraph("Multi-agent systematic equity strategy", TITLE),
        Paragraph(
            f"Phase 0 build report &mdash; steps 1 to 5 of 6 &mdash; {date.today():%d %B %Y}<br/>"
            "US large and mid caps, daily decision, paper trading first. "
            "Every figure below is measured by this repository&rsquo;s own harness.",
            SUB),
        HRFlowable(width="100%", thickness=0.7, color=INK, spaceAfter=12),
    ]

    s += [
        Paragraph("The finding, before the detail", H2),
        Paragraph(
            "<b>No configuration beats equal weight on risk-adjusted terms.</b> The three "
            "price factors have no cross-sectional edge at all over 2016&ndash;2024: the "
            "permutation test returns p = 0.295 while the Deflated Sharpe passes at 99.4%, "
            "so the return is real but did not come from selecting stocks. It is beta. "
            "Step 5 established that cheaper execution cannot rescue it.",
            BODY),
        Paragraph(
            "Value and Quality, tested separately once the lake held filings for the whole "
            "universe, are different: the ranking does carry information. A 20-name book "
            "passes the permutation test at p = 0.050. It still does not license trading, "
            "because the entire gain is risk rather than skill &mdash; levering equal weight "
            "to the same volatility returns +21.18% against the strategy&rsquo;s +21.44%.",
            BODY),
        Paragraph(
            "What the project did produce is a harness that catches this. It rejected its "
            "own strategy four times, and twenty-four defects found along the way were of "
            "the silent kind &mdash; a plausible answer rather than an error.",
            BODY),
    ]

    s += [Paragraph("What was built", H2)]
    s += [table(
        text_cols=(1,),
        rows=[["", "Step", "State"],
         ["1", "Core contracts: Signal, Agent, Context, asof-bounded MarketView", "done"],
         ["2", "Data layer: yfinance, SEC EDGAR, FRED to DuckDB/parquet", "done"],
         ["3", "Backtester: walk-forward engine, cost model, three validation tests", "done"],
         ["4", "Five agents: Momentum, Value, Quality, LowVol, Reversal", "done"],
         ["5", "Risk layer: hard limits with a veto, turnover control, drawdown stop", "done"],
         ["6", "Scheduling: cron to Alpaca paper", "not started"]],
        widths=[12 * mm, width - 42 * mm, 30 * mm])]
    s += [Paragraph(
        "5,728 lines of code, 3,921 lines of tests, 317 tests passing with no network access.",
        CAPTION)]

    s += [
        Paragraph("Step 4: selection does not help, at either universe size", H2),
        Paragraph(
            "Each sweep varies how many names the strategy holds, from concentrated to the "
            "whole universe. Monthly rebalance.",
            BODY),
        Paragraph("30 names, net of 10 bps per side", H3),
        table([["Names held", "5", "10", "15", "25", "all 30"],
               ["Sharpe", "0.83", "1.00", "1.08", "1.14", "1.25"]],
              [32 * mm] + [(width - 32 * mm) / 5] * 5, header=True),
        Paragraph(
            "Sharpe rises monotonically as the strategy holds more of the universe. Every "
            "step of selection cost return; the best thing the agents could do was stop "
            "selecting.", CAPTION),
        Paragraph("474 names, gross of costs", H3),
        table([["Names held", "20", "50", "100", "200", "400", "all 474"],
               ["Sharpe", "1.04", "1.06", "1.02", "1.07", "1.08", "1.07"]],
              [32 * mm] + [(width - 32 * mm) / 6] * 6, header=True),
        Paragraph(
            "A larger universe removed the harm but put nothing in its place: the sweep is "
            "flat, and the strategy gross of costs equals equal weight within noise. Net of "
            "10 bps it loses, for the dullest reason &mdash; turnover is nonzero and the "
            "gross edge is zero.", CAPTION),
    ]

    s += [
        Paragraph("Step 5: a turnover budget de-levers, it does not improve", H2),
        Paragraph(
            "The obvious fix was to trade less. A no-trade band and rank hysteresis barely "
            "bit, because the turnover is membership churn rather than drift. A turnover "
            "budget did cut costs &mdash; by holding cash.", BODY),
        table([["Monthly budget", "Mean gross", "Sharpe", "CAGR"],
               ["0.20", "0.375", "1.06", "+6.0%"],
               ["0.35", "0.558", "1.08", "+9.4%"],
               ["0.50", "0.750", "1.07", "+12.7%"],
               ["0.75", "0.964", "0.96", "+15.1%"],
               ["1.00", "0.985", "0.96", "+15.7%"]],
              [40 * mm, 35 * mm, 30 * mm, width - 105 * mm]),
        Paragraph(
            "Sharpe is flat while CAGR more than doubles, and mean gross exposure tracks "
            "CAGR exactly. At the budget that keeps the book invested, Sharpe is back to the "
            "unconstrained 0.96. There is no budget that keeps the signal and loses the cost: "
            "this signal&rsquo;s information decays about as fast as its membership churns.",
            CAPTION),
        Spacer(1, 7),
        Paragraph(
            "<b>Sharpe concealed this.</b> The 0.20 run reported Sharpe 1.06 &mdash; matching "
            "fully-invested equal weight &mdash; on a book 37.5% in cash, with a 14% maximum "
            "drawdown against 38%. Sharpe is scale-invariant to cash, so it cannot distinguish "
            "a strategy that will not deploy capital from one that does. Mean gross exposure "
            "is now reported beside it and warns below 90%.", BODY),
    ]

    s += [
        Paragraph("Validation: every price-factor run reached the same verdict", H2),
        table([["Universe", "Observed", "Random mean", "p-value", "Deflated Sharpe", "Verdict"],
               ["30 names, top-25", "1.14", "1.16", "0.579", "99.6%", "FAIL"],
               ["474 names, top-50", "0.96", "0.93", "0.295", "99.4%", "FAIL"]],
              [38 * mm, 22 * mm, 26 * mm, 20 * mm, 30 * mm, width - 136 * mm]),
        Paragraph(
            "The permutation test keeps every recorded decision &mdash; same dates, same "
            "position count, same weights &mdash; and hands each day&rsquo;s weights to random "
            "tradeable names. Random portfolios of the same shape did as well. Read with the "
            "Deflated Sharpe, which passed: the return is real and it is not from selection.",
            CAPTION),
    ]

    if part == "a":
        doc.build(
            s,
            onFirstPage=lambda c, d: footer(c, d, page_offset),
            onLaterPages=lambda c, d: footer(c, d, page_offset),
        )
        return out

    s = []
    s += [
        Paragraph("Value and Quality, once the filings existed", H2),
        Paragraph(
            "886k XBRL facts for 499 of 500 filers, so Value scores 326&ndash;383 names per "
            "date and Quality 234&ndash;272, against 20 before. Same window, same costs.",
            BODY),
        table([["top-N held", "20", "50", "100", "400", "equal weight"],
               ["Sharpe", "1.07", "1.04", "1.02", "0.97", "1.06"],
               ["permutation p", "0.050", "0.055", "0.090", "0.697", "—"],
               ["turnover/day", "0.69%", "0.58%", "0.49%", "0.39%", "0.16%"]],
              widths=[34 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, width - 122 * mm]),
        Paragraph(
            "This sweep is not flat, and that is the finding. Sharpe falls and the p-value "
            "rises monotonically as the book widens, no point out of order &mdash; four "
            "points ordering themselves by chance is a 1-in-24 event. The top of the ranking "
            "really is better than the bottom, which the price factors never showed.",
            CAPTION),
        Spacer(1, 7),
        Paragraph(
            "<b>It still does not license trading it.</b> The top-20 book is the first PASS "
            "in this project, at p = 0.050 and a Deflated Sharpe of 99.8%. Four reasons that "
            "is not enough: the whole gain is risk rather than skill &mdash; at 1.079x equal "
            "weight&rsquo;s volatility, levering equal weight to match returns +21.18% "
            "against +21.44%, leaving 0.26pp to selection and a Sharpe gap of 0.01 inside a "
            "CI of [0.36, 1.81]; p = 0.050 is the boundary and four configurations were "
            "tried, which clears 5% by luck alone 18.5% of the time; the permutation null is "
            "random <i>20-name</i> books at Sharpe 0.89, not the 1.06 equal-weight book it "
            "merely ties while paying 4.3x the turnover; and survivorship bias selects on "
            "exactly the variable a value factor trades, because the cheap names still in "
            "today&rsquo;s top 500 are the ones that recovered.", BODY),
    ]

    s += [
        Paragraph("Known biases in the data", H2),
        Paragraph(
            "These are limitations of a zero-cost data budget, not defects. They are reprinted "
            "by <font face='Courier'>hf status</font> so they cannot be quietly forgotten.",
            BODY),
        table(text_cols=(1,), rows=[["Bias", "Effect"],
               ["Survivorship and selection",
                "Universe is the top 500 by today's market cap, backtested from 2016, which "
                "selects the companies that grew into it. Absolute returns are inflated for "
                "every strategy including equal weight; the comparison between them survives."],
               ["Retroactive price adjustment",
                "adj_close is adjusted as of today, correct for returns and wrong for price "
                "levels. A split between a filing and the asof is wrong by a whole-number "
                "factor and is not correctable from this data."],
               ["Revised macro series",
                "Treasury yields and VIX are never revised and are point-in-time correct. "
                "CPI and unemployment are, so a lake holding them holds a cleaner series "
                "than anyone had live. None are ingested yet, and hf status names only the "
                "series actually present."],
               ["Dual-currency filers",
                "43 of 500 file the same fact in their home currency and in USD. Each "
                "series is narrowed to one currency, and a ratio across two known-different "
                "units is refused outright: there is no FX series here to convert with."]],
              widths=[42 * mm, width - 42 * mm]),
    ]

    s += [
        Paragraph("Defects found, and what caught them", H2),
        table(text_cols=(2,), rows=[["Found in", "Count", "Character"],
               ["Build (steps 1-5)", "8", "Mostly caught by tests within minutes of writing"],
               ["Code review", "15", "All fixed; three were silent and flattering"],
               ["First contact with real EDGAR data", "5", "None were caught by 8 synthetic-payload tests"],
               ["Scaling that ingest to 500 filers", "6", "Four would have changed a reported number"]],
              widths=[52 * mm, 18 * mm, width - 70 * mm]),
        Paragraph(
            "The bottom two rows are the ones worth keeping. Eight synthetic-payload tests "
            "all passed, and the first real request exposed five defects. Scaling from 20 "
            "filers to 500 exposed six more, none of which announced itself: a ratio built "
            "across two currencies read a book-to-price of 18.05 against a true 0.29, and a "
            "61-day smoke test pooled into the Deflated Sharpe raised the multiple-testing "
            "hurdle twentyfold and failed a run that passes.", CAPTION),
    ]

    s += [
        Paragraph("What remains open", H2),
        Paragraph(
            "<b>Only one of the four doubts about top-20 can be removed.</b> Three are about "
            "effect size and will not move. The fourth, survivorship bias, is a measurement "
            "problem with a known answer: point-in-time universe membership, which is a paid "
            "dataset. Running more configurations against the same biased universe would "
            "only raise what the Deflated Sharpe charges.", BODY),
        Paragraph(
            "<b>Step 6 is deliberately not built.</b> Scheduling a book that ties equal "
            "weight while paying 4.3x the turnover is not a thing to automate.", BODY),
        Paragraph(
            "<b>Market cap is wrong for names that split between a filing and the asof.</b> "
            "The share count comes from the filing and the price from the tape, with nothing "
            "reconciling a split in between &mdash; a whole-number error. Fixing it needs "
            "split events in the data layer.", BODY),
        Paragraph(
            "<b>The trial log holds 20 entries on this window.</b> Every further "
            "configuration raises what the Deflated Sharpe charges, so the next test should "
            "answer a question rather than search for a flattering number.", BODY),
    ]

    doc.build(
        s,
        onFirstPage=lambda c, d: footer(c, d, page_offset),
        onLaterPages=lambda c, d: footer(c, d, page_offset),
    )
    return out


def build_diagram_page(page_offset: int = 0) -> pathlib.Path | None:
    """The diagram on its own landscape page.

    A 1.7:1 drawing squeezed into a portrait column renders its labels at
    about four pixels. Landscape gives it the full page and keeps them
    readable, which is the only reason to include a diagram at all.
    """
    if not (ROOT / "docs" / "decision-logic.png").exists():
        return None
    out = BUILD / "_diagram-page.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=landscape(A4),
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=16 * mm,
    )
    width = doc.width
    s = [
        Paragraph("The decision logic", H2),
        Paragraph(
            "Organised around the points at which the system refuses, because those are what "
            "the design is about. Agents may only emit a Signal; everything from there to a "
            "weight is deterministic arithmetic; and NAV &mdash; hence the drawdown stop "
            "&mdash; belongs to the engine, which in live trading is the broker, not the model.",
            BODY),
    ]
    caption = Paragraph(
        "Four refusal points, each meaning something different: an agent omits a ticker it "
        "cannot evaluate (distinct from scoring it zero); the optimizer returns nothing when no "
        "name clears its threshold; the risk manager holds cash rather than padding a thin book "
        "with names the strategy never chose; and the stop flattens regardless of what the "
        "strategy asked for.", CAPTION)

    drawing = diagram(width)
    if drawing is not None:
        # Give the image whatever the text does not use, measured rather than
        # guessed: a reserve that is too small pushes the caption onto a second
        # page, which the merge then has to refuse.
        gap = 6
        used = sum(f.wrap(width, doc.height)[1] + f.getSpaceBefore()
                   + f.getSpaceAfter() for f in (*s, caption))
        max_h = doc.height - used - gap
        if drawing.drawHeight > max_h:
            scale = max_h / drawing.drawHeight
            drawing.drawWidth *= scale
            drawing.drawHeight *= scale
        s += [Spacer(1, 2), drawing]
    s += [caption]
    doc.build(
        s,
        onFirstPage=lambda c, d: footer(c, d, page_offset),
        onLaterPages=lambda c, d: footer(c, d, page_offset),
    )
    return out


def assemble() -> pathlib.Path:
    """Part A, the landscape plate, then part B — concatenated, not spliced.

    Each piece is its own document, so page numbering is continuous only
    because every piece is told the offset it lands at. That is the price of
    not letting a paragraph straddle a landscape page.
    """
    from pypdf import PdfReader, PdfWriter

    plate = build_diagram_page(page_offset=0)  # offset fixed up below

    part_a = build("a", page_offset=0)
    a_pages = PdfReader(part_a).pages
    n_a = len(a_pages)

    if plate is not None:
        # Rebuild the plate now that its page number is known.
        plate.unlink(missing_ok=True)
        plate = build_diagram_page(page_offset=n_a)
        plate_pages = PdfReader(plate).pages
        if len(plate_pages) != 1:
            raise RuntimeError(
                f"The diagram plate came out as {len(plate_pages)} pages, not one. "
                "Taking only the first would drop its caption without saying so; "
                "shrink the image instead.")

    offset_b = n_a + (1 if plate is not None else 0)
    part_b = build("b", page_offset=offset_b)

    final = BUILD / "hedge-fund-phase0-report.pdf"
    writer = PdfWriter()
    for page in a_pages:
        writer.add_page(page)
    if plate is not None:
        writer.add_page(PdfReader(plate).pages[0])
    for page in PdfReader(part_b).pages:
        writer.add_page(page)
    writer.add_metadata({
        "/Title": "hedge_fund - Phase 0 report",
        "/Author": "Chaut007",
        "/Subject": "Systematic equity strategy: build and findings",
    })
    with final.open("wb") as fh:
        writer.write(fh)
    for scratch in (part_a, part_b, plate):
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    return final


if __name__ == "__main__":
    path = assemble()
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
