# Place the legend/key *under* the chart using Graphviz's graph label.
from pathlib import Path
from graphviz import Digraph

LANES = [
    ("Ingest",        {"bg": "#E8F1FA", "node": "#90CAF9"}),
    ("Pre/Post",      {"bg": "#EEF7EE", "node": "#A5D6A7"}),
    ("Model",         {"bg": "#FFF4E6", "node": "#FFCC80"}),
    ("Detection",     {"bg": "#F1E8FF", "node": "#B39DDB"}),
    ("QA",            {"bg": "#FFFBEA", "node": "#FFE082"}),
    ("Output",        {"bg": "#E7F6F3", "node": "#80CBC4"}),
]

MAIN_EDGE = "#263238"
SIDE_EDGE = "#D81B60"

def make_graph_with_bottom_legend(stem: str, fmt: str = "png"):
    dot = Digraph("data_flow_swimlanes_v4", format=fmt, engine="dot")
    dot.attr(rankdir="LR", splines="ortho", nodesep="0.95", ranksep="1.05", dpi="700", margin="0.05", concentrate="false")
    dot.attr("node", shape="box", style="rounded,filled", color="#333333", fontname="Helvetica", fontsize="12", penwidth="1.4")
    dot.attr("edge", arrowsize="0.9", penwidth="1.4")

    # --- Swimlanes (clusters) ---
    with dot.subgraph(name="cluster_ingest") as c:
        c.attr(label="Ingest", labelloc="t", color="#B0C8E8", fontsize="13", fontname="Helvetica-Bold",
               style="filled", bgcolor=LANES[0][1]["bg"], penwidth="1.2")
        c.node("pds", "PDS Ldr", fillcolor=LANES[0][1]["node"])
        c.node("mdl", "Model Ldr\n(wts/cfg)", fillcolor=LANES[0][1]["node"])

    with dot.subgraph(name="cluster_prepost") as c:
        c.attr(label="Pre/Post", labelloc="t", color="#B9DDB7", fontsize="13", fontname="Helvetica-Bold",
               style="filled", bgcolor=LANES[1][1]["bg"], penwidth="1.2")
        c.node("pre", "Preproc", fillcolor=LANES[1][1]["node"])
        c.node("pp",  "Postproc", fillcolor=LANES[1][1]["node"])

    with dot.subgraph(name="cluster_model") as c:
        c.attr(label="Model", labelloc="t", color="#FFD2A3", fontsize="13", fontname="Helvetica-Bold",
               style="filled", bgcolor=LANES[2][1]["bg"], penwidth="1.2")
        c.node("bim", "BIM", fillcolor=LANES[2][1]["node"])

    with dot.subgraph(name="cluster_detect") as c:
        c.attr(label="Detection", labelloc="t", color="#D9CBFF", fontsize="13", fontname="Helvetica-Bold",
               style="filled", bgcolor=LANES[3][1]["bg"], penwidth="1.2")
        c.node("fd", "Feat Detect", fillcolor=LANES[3][1]["node"])

    with dot.subgraph(name="cluster_qa") as c:
        c.attr(label="QA / Review", labelloc="t", color="#FFE9A8", fontsize="13", fontname="Helvetica-Bold",
               style="filled", bgcolor=LANES[4][1]["bg"], penwidth="1.2")
        c.node("rev", "Review", fillcolor=LANES[4][1]["node"])

    with dot.subgraph(name="cluster_output") as c:
        c.attr(label="Output", labelloc="t", color="#A6E3D8", fontsize="13", fontname="Helvetica-Bold",
               style="filled", bgcolor=LANES[5][1]["bg"], penwidth="1.2")
        c.node("exp", "Export", fillcolor=LANES[5][1]["node"])

    # --- Edges ---
    # Main flow
    dot.edge("pds", "pre", color=MAIN_EDGE)
    dot.edge("pre", "bim", color=MAIN_EDGE)
    dot.edge("mdl", "bim", color=MAIN_EDGE)  # second input to BIM
    dot.edge("bim", "fd", color=MAIN_EDGE)
    dot.edge("fd",  "rev", color=MAIN_EDGE)
    dot.edge("rev", "exp", color=MAIN_EDGE)
    # Side branch
    dot.edge("pds", "pp", color=SIDE_EDGE)
    dot.edge("pp", "fd", color=SIDE_EDGE)

    # --- Bottom legend via graph label ---
    legend_html = r"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="2" BGCOLOR="#F7F7F7"><B>Key</B></TD></TR>
    <TR><TD WIDTH="22" BGCOLOR="#90CAF9"></TD><TD>Ingest (PDS Ldr, Model Ldr)</TD></TR>
    <TR><TD WIDTH="22" BGCOLOR="#A5D6A7"></TD><TD>Pre/Post (Preproc, Postproc)</TD></TR>
    <TR><TD WIDTH="22" BGCOLOR="#FFCC80"></TD><TD>Model (BIM)</TD></TR>
    <TR><TD WIDTH="22" BGCOLOR="#B39DDB"></TD><TD>Detection (Feat Detect)</TD></TR>
    <TR><TD WIDTH="22" BGCOLOR="#FFE082"></TD><TD>QA / Review</TD></TR>
    <TR><TD WIDTH="22" BGCOLOR="#80CBC4"></TD><TD>Output (Export)</TD></TR>
    <TR><TD WIDTH="22"><FONT COLOR="#263238">——</FONT></TD><TD>Main Flow</TD></TR>
    <TR><TD WIDTH="22"><FONT COLOR="#D81B60">——</FONT></TD><TD>Side Branch (PDS → Postproc → Feat Detect)</TD></TR>
    </TABLE>>"""
    dot.attr(label=legend_html, labelloc="b", labeljust="c")

    out = Path(stem)
    dot.render(out, cleanup=True)
    return f"{out}.{fmt}"

png = make_graph_with_bottom_legend("/mnt/data/data_flow_swimlanes_v4", "png")
svg = make_graph_with_bottom_legend("/mnt/data/data_flow_swimlanes_v4", "svg")
png, svg
