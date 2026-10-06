# Student Performance Intelligence

A professional Streamlit dashboard for analyzing student marks and attendance with **LangChain + Groq**. It accepts CSV and Excel files, calculates statistics locally, and uses short, controlled Groq calls to produce clear academic insights.

## Product overview

The dashboard is designed for a teacher or academic coordinator who needs a fast class snapshot:

- Upload a `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, `.json`, `.parquet`, `.pdf`, or `.docx` file.
- Preview cleaned and validated data before analysis.
- See class average, attendance, top performer, and students needing attention.
- Generate student-level evaluations and class comparisons.
- Read an executive class report with strengths and recommended interventions.
- Download the complete report as JSON or student insights as CSV.

### Supported input layouts

```text
student, math, science, english, attendance
```

The Intake Agent understands common alternatives such as `Student Name`, `Name`, `Mathematics`, `Physics`, `Language`, `Present`, and `Attendance Rate`. It also detects sheets where metrics are arranged as rows and student names are arranged as columns. PDF and DOCX inputs must contain a real table. If a file contains no analyzable numeric performance fields, the app explains what is missing instead of silently producing a bad report. If attendance is not present, it keeps the dataset usable and clearly marks attendance as 100% rather than blocking the upload. Scores and attendance must be between 0 and 100.

## Four-agent architecture

The interface presents seven focused stages so the user can see exactly what is happening. This keeps behavior predictable and controls token usage.

1. **Intake Agent** — understands headers, rows, columns, and data types locally.
2. **Calculation Engine** — computes grades, averages, ranks, and trends locally.
3. **Performance Agent** — one short Groq call per student for level and strengths.
4. **Benchmark Agent** — local gap calculation plus one short Groq call per student.
5. **Recommendation Agent** — turns performance findings into next steps.
6. **Report Agent** — one Groq call for the class summary and interventions.
7. **Quality & Export Agent** — checks the result and prepares JSON/CSV downloads.

All arithmetic is local. The LLM receives compact records instead of the full spreadsheet on every call.

## Token controls

Configure strict output limits in `.env` or your deployment secrets. The website intentionally has **no API-key field and no Groq configuration controls**; credentials stay server-side.

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
SCHEMA_MAX_TOKENS=180
EVAL_MAX_TOKENS=120
COMPARE_MAX_TOKENS=120
REPORT_MAX_TOKENS=220
```

The dashboard also limits prompt output length and requests JSON-only responses. Total calls for `N` students are `N + N + 1`.

After a run, the dashboard shows an **Agent Activity Report** in order. Each stage explains what it did, how many records it processed, which columns it understood, what statistics it calculated, and what output it produced. The results also include best performers, priority support learners, a learner-average-versus-class-average diagram, and a subject score chart.

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-intelligence.git
cd student-performance-intelligence
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env
# Add GROQ_API_KEY to .env
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Streamlit Community Cloud deployment

1. Push this folder to a GitHub repository.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io).
3. Select the repository and set the main file to `app.py`.
4. In **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_key"
GROQ_MODEL = "llama-3.1-8b-instant"
EVAL_MAX_TOKENS = "120"
COMPARE_MAX_TOKENS = "120"
REPORT_MAX_TOKENS = "220"
```

Never commit `.env` or API keys. `.gitignore` already excludes local secrets.

## Repository structure

```text
.
├── app.py
├── data_students.csv
├── .env.example
├── .gitignore
├── .streamlit_config.toml
├── requirements.txt
└── README.md
```

## Privacy and responsible use

The app performs calculations locally, but compact student summaries are sent to Groq when analysis is run. Use student IDs or pseudonyms where appropriate, follow your institution's data policy, and obtain the required permission before uploading identifiable student records. The output is decision support and should be reviewed by a qualified teacher; it should not be the sole basis for an educational decision.
