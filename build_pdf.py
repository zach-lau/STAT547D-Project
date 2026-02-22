""" Modified Chat GPT code """
# This script scans the current working directory for experiment folders.
# For each experiment, it collects trace.png and pairs.png from each algorithm subfolder,
# and creates two PDFs:
#   1) trace_plots.pdf  (2x2 grid per experiment of trace.png)
#   2) pairs_plots.pdf  (2x2 grid per experiment of pairs.png)

import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import utils
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER


# Root directory (change if needed)
root_dir = os.getcwd()



def get_image(path, max_width=3*inch, max_height=3*inch):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    if iw > ih:
        width = min(max_width, iw)
        height = width * aspect
    else:
        height = min(max_height, ih)
        width = height / aspect
    return Image(path, width=width, height=height)

def collect_experiments(base_dir):
    experiments = []
    seed = 123
    ns = [10, 100, 1000]
    ms = [10, 100]
    ds = [1, 5, 50]
    for d in ds:
        for m in ms:
            for n in ns:
                item = f"{n}-{d}-{m}-{seed}"
                exp_path = os.path.join(base_dir, item)
                if os.path.isdir(exp_path):
                    experiments.append(exp_path)
    return experiments

def build_pdf(output_path, image_name):
    doc = SimpleDocTemplate(output_path)
    elements = []
    styles = getSampleStyleSheet()
    centered_style = ParagraphStyle(
        name="CenteredHeading",
        parent=styles["Heading4"],
        alignment=TA_CENTER,
    )

    experiments = collect_experiments(root_dir)

    for idx, exp_path in enumerate(experiments):
        exp_name = os.path.basename(exp_path)
        elements.append(Paragraph(f"Experiment: {exp_name}", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        alg_dirs = [
            os.path.join(exp_path, d)
            for d in sorted(os.listdir(exp_path))
            if os.path.isdir(os.path.join(exp_path, d))
        ]

        labels = ["Adjusted Mean", "True LML", "Langevin", "Mean"]
        blocks = []
        for i, alg_dir in enumerate(alg_dirs):
            img_path = os.path.join(alg_dir, image_name)
            if os.path.exists(img_path):
                img = get_image(img_path)
            else:
                img = Paragraph("Missing", styles["Normal"])
            title = Paragraph(labels[i], centered_style)
            blocks.append([title, Spacer(1, 0.1 * inch), img])

        # Ensure 4 slots (2x2)
        while len(blocks) < 4:
            blocks.append("")

        grid = [
            [blocks[0], blocks[1]],
            [blocks[2], blocks[3]]
        ]

        table = Table(grid, colWidths=[3.2*inch]*2)

        table.setStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])

        elements.append(table)
        # elements.append(Spacer(1, 0.5 * inch))
        if idx < len(experiments) - 1:
            elements.append(PageBreak())

    doc.build(elements)

# Output files
trace_pdf_path = "trace_plots.pdf"
pairs_pdf_path = "pairs_plots.pdf"

build_pdf(trace_pdf_path, "trace.png")
build_pdf(pairs_pdf_path, "pairs.png")

trace_pdf_path, pairs_pdf_path
