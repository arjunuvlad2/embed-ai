"""
Same fictional "NorthStar Analytics" knowledge base from the Day 1 RAG
exercise — reused deliberately, so it's obvious that RAG (retrieval) is
itself one specific technique for context engineering, not a separate topic.
"""

DOCUMENTS = [
    "NorthStar Analytics employees accrue 18 days of paid time off (PTO) per "
    "calendar year, prorated for new hires based on start date. PTO requests "
    "must be submitted in the HR portal at least 5 business days in advance "
    "for approval by your manager.",

    "Remote work is permitted up to 3 days per week for all full-time "
    "employees. Fully remote arrangements require director-level approval "
    "and are reviewed quarterly. Employees working from outside their home "
    "country for more than 30 consecutive days must notify HR for tax and "
    "compliance reasons.",

    "Expense reimbursements under $75 do not require pre-approval and can be "
    "submitted directly through the Expensify app with a receipt. Expenses "
    "over $75, including client travel and equipment purchases, require "
    "manager sign-off before the expense is incurred.",

    "New hires complete a 2-week onboarding program covering company "
    "systems, security training, and a shadowing period with their team "
    "lead. IT equipment (laptop, monitor, headset) is shipped to arrive on "
    "or before the employee's first day.",

    "All code changes require at least one peer review approval before "
    "merging to the main branch. Pull requests should include a description "
    "of the change, testing performed, and a link to the relevant ticket. "
    "Critical production hotfixes may bypass review with post-hoc sign-off "
    "from the engineering lead.",

    "In the event of a production incident, the on-call engineer must "
    "acknowledge the page within 15 minutes and post an initial status "
    "update in the #incidents Slack channel within 30 minutes. A "
    "post-incident review is required for any outage lasting longer than "
    "1 hour.",

    "Performance reviews are conducted twice a year, in June and December. "
    "Employees complete a self-assessment, which is then reviewed alongside "
    "manager and peer feedback. Promotion decisions are finalized in the "
    "December review cycle.",

    "All client-facing communications, including emails and deliverables, "
    "must be reviewed by a project lead before sending during an employee's "
    "first 90 days. After that period, senior consultants may communicate "
    "directly with clients within their engagement scope.",

    "Company laptops must have disk encryption and the corporate VPN client "
    "installed before connecting to any internal systems. Employees are "
    "prohibited from storing client data on personal devices or unapproved "
    "cloud storage services.",

    "Equipment requests for non-standard hardware (e.g., a second monitor, "
    "ergonomic keyboard, or standing desk) can be submitted through the IT "
    "portal and are typically fulfilled within 5 business days, subject to "
    "budget approval from your department head.",
]
