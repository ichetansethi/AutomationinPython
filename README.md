# 🐞 Bug Tracker Dashboard

An interactive dashboard was built using **Streamlit** to help teams track and analyze issues by category, severity, and status. Designed with clean visuals, flexible filters, and email-ready summaries for weekly and monthly client reporting.

---

## 📸 Preview

![Bug Tracker Dashboard](./assets/dashboard_screenshot.png)

---

## ✨ Features

- **📂 Issues by Category**  
  View total issues and average resolution time categorized by source (e.g., Data, Understanding, Bug).

- **🚦 Issues by Severity**  
  Highlights count and resolution trends for P1, P2, P3, and P4 issues using intuitive color coding.

- **✅ Status Filters**  
  Filter issues dynamically with **checkboxes** for Open, Pending, and In Progress.

- **🎨 Color Coding for Insights**  
  - Custom gradients for resolution times (blue scale)
  - Severity & category-based coloring for issue counts

- **📬 Weekly & Monthly Email Templates**  
  Auto-generated, non-technical emails with embedded dashboard screenshots and positive insights.

- **🖼️ Screenshot-ready Design**  
  Layout structured for easy screen capture and sharing with clients.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **Custom Styling with Pandas Styler**
- *(Optional: Seaborn or Matplotlib for graphs)*

---

## 🚀 Getting Started

### 🔧 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/bug-tracker-dashboard.git
cd bug-tracker-dashboard

Create a virtual environment and install dependencies:

pip install -r requirements.txt

▶️ Run the App

streamlit run dashboard.py



⸻

🗂️ Folder Structure

📦 bug-tracker-dashboard
├── dashboard.py
├── requirements.txt
├── README.md
├── assets/
│   └── dashboard_screenshot.png
└── utils/
    └── filters.py



⸻

📧 Email Reporting

The dashboard supports automated weekly and monthly reporting with built-in email templates. Just embed a screenshot in the email body using the format provided in the templates/ folder.

⸻

🙌 Contributions

Just to let you know, pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change or improve.

⸻

📄 License

MIT
