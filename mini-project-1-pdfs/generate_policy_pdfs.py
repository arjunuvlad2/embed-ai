"""
Generates 10 simple, one-page PDF policy documents for Mini Project 1 —
the same NorthStar Analytics content from rag-search-exercise/knowledge_base.py,
as real PDF files instead of Python strings. This is the only thing that
changes for the assignment: students extract text from these PDFs instead
of importing a list of strings, then the rest of the RAG pipeline is
identical to rag_search.py.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

NAVY = colors.HexColor("#12163B")
TEAL = colors.HexColor("#17B8BD")
GRAY = colors.HexColor("#6B7280")
INK = colors.HexColor("#1A1F36")

kicker_style = ParagraphStyle(
    "Kicker", fontName="Helvetica-Bold", fontSize=9.5, textColor=TEAL,
    spaceAfter=4, leading=12,
)
title_style = ParagraphStyle(
    "Title", fontName="Times-Bold", fontSize=20, textColor=NAVY,
    spaceAfter=14, leading=24,
)
body_style = ParagraphStyle(
    "Body", fontName="Helvetica", fontSize=11.5, textColor=INK, leading=17,
)
footer_style = ParagraphStyle(
    "Footer", fontName="Helvetica-Oblique", fontSize=9, textColor=GRAY,
)

DOCUMENTS = [
    (
        "01-time-off-policy",
        "Time Off (PTO) Policy",
        "NorthStar Analytics employees accrue 18 days of paid time off (PTO) per "
        "calendar year, prorated for new hires based on start date. PTO requests "
        "must be submitted in the HR portal at least 5 business days in advance "
        "for approval by your manager.",
    ),
    (
        "02-remote-work-policy",
        "Remote Work Policy",
        "Remote work is permitted up to 3 days per week for all full-time "
        "employees. Fully remote arrangements require director-level approval "
        "and are reviewed quarterly. Employees working from outside their home "
        "country for more than 30 consecutive days must notify HR for tax and "
        "compliance reasons.",
    ),
    (
        "03-expense-reimbursement-policy",
        "Expense Reimbursement Policy",
        "Expense reimbursements under $75 do not require pre-approval and can be "
        "submitted directly through the Expensify app with a receipt. Expenses "
        "over $75, including client travel and equipment purchases, require "
        "manager sign-off before the expense is incurred.",
    ),
    (
        "04-new-hire-onboarding",
        "New Hire Onboarding",
        "New hires complete a 2-week onboarding program covering company "
        "systems, security training, and a shadowing period with their team "
        "lead. IT equipment (laptop, monitor, headset) is shipped to arrive on "
        "or before the employee's first day.",
    ),
    (
        "05-code-review-policy",
        "Code Review Policy",
        "All code changes require at least one peer review approval before "
        "merging to the main branch. Pull requests should include a description "
        "of the change, testing performed, and a link to the relevant ticket. "
        "Critical production hotfixes may bypass review with post-hoc sign-off "
        "from the engineering lead.",
    ),
    (
        "06-incident-response-policy",
        "Incident Response Policy",
        "In the event of a production incident, the on-call engineer must "
        "acknowledge the page within 15 minutes and post an initial status "
        "update in the #incidents Slack channel within 30 minutes. A "
        "post-incident review is required for any outage lasting longer than "
        "1 hour.",
    ),
    (
        "07-performance-review-policy",
        "Performance Review Policy",
        "Performance reviews are conducted twice a year, in June and December. "
        "Employees complete a self-assessment, which is then reviewed alongside "
        "manager and peer feedback. Promotion decisions are finalized in the "
        "December review cycle.",
    ),
    (
        "08-client-communication-policy",
        "Client Communication Policy",
        "All client-facing communications, including emails and deliverables, "
        "must be reviewed by a project lead before sending during an employee's "
        "first 90 days. After that period, senior consultants may communicate "
        "directly with clients within their engagement scope.",
    ),
    (
        "09-data-security-policy",
        "Data Security Policy",
        "Company laptops must have disk encryption and the corporate VPN client "
        "installed before connecting to any internal systems. Employees are "
        "prohibited from storing client data on personal devices or unapproved "
        "cloud storage services.",
    ),
    (
        "10-equipment-request-policy",
        "Equipment Request Policy",
        "Equipment requests for non-standard hardware (e.g., a second monitor, "
        "ergonomic keyboard, or standing desk) can be submitted through the IT "
        "portal and are typically fulfilled within 5 business days, subject to "
        "budget approval from your department head.",
    ),
]


def build_pdf(filename, title, body_text):
    doc = SimpleDocTemplate(
        filename, pagesize=letter,
        leftMargin=1 * inch, rightMargin=1 * inch,
        topMargin=1 * inch, bottomMargin=1 * inch,
    )
    story = [
        Paragraph("NORTHSTAR ANALYTICS", kicker_style),
        Paragraph(title, title_style),
        HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=16),
        Paragraph(body_text, body_style),
        Spacer(1, 40),
        Paragraph("Internal Policy Document — Page 1 of 1", footer_style),
    ]
    doc.build(story)


def main():
    for slug, title, body_text in DOCUMENTS:
        filename = f"{slug}.pdf"
        build_pdf(filename, title, body_text)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
