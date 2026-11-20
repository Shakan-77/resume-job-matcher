"""Skill extraction with a curated vocabulary.

Each skill has a canonical name and a few aliases ("JavaScript": "js",
"ECMAScript", ...). ``extract_skills`` finds them with whole-word regexes.
Fuzzy, meaning-based matching (e.g. "PyTorch" ~ "deep learning frameworks")
happens later in ``scoring.py`` using embeddings.
"""

from __future__ import annotations

import re
from functools import lru_cache

# category -> {canonical skill name: [aliases]}
# The canonical name itself is always searched for, so only list *extra* spellings.
SKILL_VOCAB: dict[str, dict[str, list[str]]] = {
    "Programming languages": {
        "Python": [],
        "Java": [],
        "JavaScript": ["js", "ecmascript"],
        "TypeScript": ["ts"],
        "C++": ["cpp"],
        "C#": ["c sharp", "csharp"],
        "C": [],
        "Go": ["golang"],
        "Rust": [],
        "Ruby": [],
        "PHP": [],
        "Kotlin": [],
        "Swift": [],
        "Scala": [],
        "R": [],
        "MATLAB": [],
        "Perl": [],
        "Bash": ["shell scripting", "shell script", "unix shell"],
        "PowerShell": [],
        "SQL": ["t-sql", "pl/sql", "plsql"],
        "Dart": [],
        "Objective-C": [],
        "VBA": [],
        "COBOL": [],
        "Assembly": [],
    },
    "Web development": {
        "HTML": ["html5"],
        "CSS": ["css3"],
        "React": ["react.js", "reactjs"],
        "Angular": ["angularjs", "angular.js"],
        "Vue.js": ["vue", "vuejs"],
        "Next.js": ["nextjs"],
        "Node.js": ["node", "nodejs"],
        "Express.js": ["express", "expressjs"],
        "Django": [],
        "Flask": [],
        "FastAPI": [],
        "Spring Boot": ["spring", "spring framework"],
        "ASP.NET": [".net", "dotnet", ".net core"],
        "Ruby on Rails": ["rails"],
        "Laravel": [],
        "jQuery": [],
        "Bootstrap": [],
        "Tailwind CSS": ["tailwind"],
        "Redux": [],
        "GraphQL": [],
        "REST APIs": ["rest", "restful", "rest api", "restful api", "restful apis", "web services"],
        "WordPress": [],
    },
    "Data & databases": {
        "PostgreSQL": ["postgres"],
        "MySQL": [],
        "SQL Server": ["mssql", "microsoft sql server"],
        "Oracle Database": ["oracle db", "oracle"],
        "SQLite": [],
        "MongoDB": ["mongo"],
        "Redis": [],
        "Cassandra": [],
        "Elasticsearch": ["elastic search", "elk"],
        "DynamoDB": [],
        "Snowflake": [],
        "BigQuery": [],
        "Redshift": [],
        "Databricks": [],
        "Apache Spark": ["spark", "pyspark"],
        "Hadoop": ["hdfs", "mapreduce"],
        "Hive": [],
        "Kafka": ["apache kafka"],
        "Airflow": ["apache airflow"],
        "dbt": [],
        "ETL": ["elt", "data pipelines", "data pipeline"],
        "Data warehousing": ["data warehouse", "data warehouses"],
        "Data modeling": ["data modelling"],
        "Pandas": [],
        "NumPy": [],
        "Excel": ["microsoft excel", "ms excel", "spreadsheets"],
        "Tableau": [],
        "Power BI": ["powerbi"],
        "Looker": [],
        "Data visualization": ["data visualisation", "dashboards", "dashboarding"],
        "Data analysis": ["data analytics", "analytics"],
        "Statistics": ["statistical analysis", "statistical modeling", "hypothesis testing"],
        "A/B testing": ["ab testing", "experimentation"],
    },
    "Machine learning & AI": {
        "Machine learning": ["ml"],
        "Deep learning": ["neural networks", "neural network"],
        "Natural language processing": ["nlp"],
        "Computer vision": ["image processing"],
        "Large language models": ["llm", "llms", "generative ai", "genai"],
        "Transformers": ["hugging face", "huggingface", "bert", "gpt"],
        "PyTorch": ["torch"],
        "TensorFlow": ["tf"],
        "Keras": [],
        "scikit-learn": ["sklearn", "scikit learn"],
        "XGBoost": ["lightgbm", "gradient boosting"],
        "OpenCV": [],
        "MLOps": ["model deployment"],
        "MLflow": [],
        "Recommender systems": ["recommendation systems", "recommendation engine"],
        "Time series forecasting": ["time series", "forecasting"],
        "Reinforcement learning": [],
        "Feature engineering": [],
        "RAG": ["retrieval augmented generation", "vector databases", "vector database"],
        "LangChain": [],
    },
    "Cloud & DevOps": {
        "AWS": ["amazon web services", "ec2", "s3", "lambda"],
        "Azure": ["microsoft azure"],
        "Google Cloud": ["gcp", "google cloud platform"],
        "Docker": ["containers", "containerization"],
        "Kubernetes": ["k8s", "eks", "aks", "gke"],
        "Terraform": ["infrastructure as code", "iac"],
        "Ansible": [],
        "Jenkins": [],
        "CI/CD": ["ci / cd", "continuous integration", "continuous delivery", "continuous deployment", "github actions", "gitlab ci"],
        "Git": ["github", "gitlab", "bitbucket", "version control"],
        "Linux": ["unix", "ubuntu", "red hat", "rhel"],
        "Microservices": ["microservice", "microservices architecture"],
        "Serverless": [],
        "Monitoring": ["prometheus", "grafana", "datadog", "splunk", "observability"],
        "Networking": ["tcp/ip", "dns", "lan", "wan", "network administration"],
        "Cybersecurity": ["information security", "network security", "cyber security"],
        "System design": ["distributed systems", "scalability", "software architecture"],
    },
    "Software engineering practices": {
        "Object-oriented programming": ["oop", "object oriented", "object-oriented design"],
        "Data structures and algorithms": ["data structures", "algorithms"],
        "Unit testing": ["test automation", "pytest", "junit", "tdd", "test-driven development"],
        "Agile": ["scrum", "kanban", "sprint planning"],
        "Jira": ["confluence"],
        "Design patterns": [],
        "Debugging": ["troubleshooting"],
        "Code review": ["code reviews"],
        "Mobile development": ["android", "ios", "react native", "flutter"],
        "Embedded systems": ["firmware", "microcontrollers", "rtos"],
        "API design": ["api development"],
        "Selenium": [],
        "QA": ["quality assurance", "manual testing"],
    },
    "Business & professional": {
        "Project management": ["pmp", "project planning", "project manager"],
        "Product management": ["product roadmap", "roadmapping", "product manager"],
        "Stakeholder management": ["stakeholders", "stakeholder communication"],
        "Communication": ["communication skills", "written communication", "verbal communication", "presentation skills", "presentations"],
        "Leadership": ["team leadership", "led a team", "people management", "mentoring", "mentored"],
        "Teamwork": ["collaboration", "cross-functional", "team player"],
        "Problem solving": ["problem-solving", "analytical skills", "critical thinking"],
        "Customer service": ["customer support", "client relations", "customer relationship"],
        "Sales": ["business development", "lead generation", "account management"],
        "Marketing": ["digital marketing", "seo", "sem", "social media marketing", "content marketing"],
        "Financial analysis": ["financial modeling", "financial modelling", "budgeting", "forecasting budgets"],
        "Accounting": ["bookkeeping", "accounts payable", "accounts receivable", "gaap", "quickbooks"],
        "CRM": ["salesforce", "hubspot"],
        "ERP": ["sap", "oracle erp", "netsuite"],
        "Business analysis": ["business analyst", "requirements gathering", "requirements analysis"],
        "Supply chain": ["logistics", "procurement", "inventory management"],
        "Human resources": ["recruiting", "recruitment", "talent acquisition", "onboarding", "payroll"],
        "Operations management": ["process improvement", "lean", "six sigma"],
        "Negotiation": [],
        "Time management": ["prioritization", "multitasking"],
        "Risk management": ["compliance", "risk assessment"],
        "UX design": ["ui/ux", "user experience", "ux", "ui design", "figma", "wireframing"],
        "Technical writing": ["documentation"],
    },
}

# Short aliases that are ordinary English words in lowercase: only match them
# when written exactly like this (case-sensitive), e.g. "Go" the language vs "go".
_CASE_SENSITIVE = {"C", "R", "Go", "TS", "TF", "ML", "Spark", "Rails", "Lambda", "Express", "Node", "Swift", "Rust",
                   "Ruby", "Assembly", "Oracle", "Lean", "Torch", "React", "Angular", "Spring", "Vue", "S3", "SAP"}
_CASED_FORM = {w.lower(): w for w in _CASE_SENSITIVE}

SKILL_CATEGORY: dict[str, str] = {skill: cat for cat, skills in SKILL_VOCAB.items() for skill in skills}
ALL_SKILLS: list[str] = list(SKILL_CATEGORY)


def _alias_pattern(alias: str) -> re.Pattern:
    """Whole-word regex for one alias. Handles symbols like C++, C#, .NET, CI/CD."""
    cased = _CASED_FORM.get(alias.lower())
    text, flags = (cased, 0) if cased else (alias, re.IGNORECASE)
    escaped = re.escape(text).replace(r"\ ", r"\s+")
    if len(text) == 1:
        # Single letters (C, R) are strict: not part of "C++", "C-level", "R&D", "Objective-C".
        return re.compile(rf"(?<![\w+#.\-]){escaped}(?![\w+#&\-])", flags)
    # Not glued to a letter/digit or to + # . (so "Java" doesn't match "JavaScript", "js" not "Node.js").
    return re.compile(rf"(?<![\w+#.]){escaped}(?![\w+#])", flags)


@lru_cache(maxsize=1)
def _compiled() -> dict[str, list[re.Pattern]]:
    """skill -> regexes for its canonical name and all aliases (built once)."""
    return {
        skill: [_alias_pattern(a) for a in [skill, *SKILL_VOCAB[SKILL_CATEGORY[skill]][skill]]]
        for skill in ALL_SKILLS
    }


def extract_skills(text: str) -> list[str]:
    """Return the canonical names of all vocabulary skills mentioned in ``text`` (in vocabulary order)."""
    return [skill for skill, patterns in _compiled().items() if any(p.search(text) for p in patterns)]


def count_skills(text: str) -> int:
    """How many distinct vocabulary skills appear in ``text`` (used to spot skill-list lines)."""
    return len(extract_skills(text))


def lines_mentioning(skill: str, lines: list[str]) -> list[str]:
    """The lines that mention a given vocabulary skill (any alias)."""
    patterns = _compiled()[skill]
    return [line for line in lines if any(p.search(line) for p in patterns)]


# A specific tool implies the broader skill: someone who lists PyTorch knows
# deep learning, someone who lists PostgreSQL knows SQL. Used to give partial
# ("related") credit with an explanation instead of reporting a false gap.
IMPLIES: dict[str, list[str]] = {
    "PyTorch": ["Deep learning", "Machine learning", "Python"],
    "TensorFlow": ["Deep learning", "Machine learning"],
    "Keras": ["Deep learning", "Machine learning"],
    "Transformers": ["Natural language processing", "Deep learning", "Machine learning"],
    "Large language models": ["Natural language processing", "Deep learning"],
    "LangChain": ["Large language models", "Python"],
    "RAG": ["Large language models"],
    "OpenCV": ["Computer vision"],
    "scikit-learn": ["Machine learning", "Python"],
    "XGBoost": ["Machine learning"],
    "Deep learning": ["Machine learning"],
    "Natural language processing": ["Machine learning"],
    "Computer vision": ["Machine learning"],
    "MLflow": ["MLOps"],
    "Pandas": ["Python", "Data analysis"],
    "NumPy": ["Python"],
    "Django": ["Python", "REST APIs"],
    "Flask": ["Python", "REST APIs"],
    "FastAPI": ["Python", "REST APIs"],
    "Spring Boot": ["Java", "REST APIs"],
    "ASP.NET": ["C#"],
    "React": ["JavaScript"],
    "Angular": ["TypeScript", "JavaScript"],
    "Vue.js": ["JavaScript"],
    "Next.js": ["React", "JavaScript"],
    "Node.js": ["JavaScript"],
    "Express.js": ["Node.js", "JavaScript"],
    "TypeScript": ["JavaScript"],
    "PostgreSQL": ["SQL"],
    "MySQL": ["SQL"],
    "SQL Server": ["SQL"],
    "Oracle Database": ["SQL"],
    "SQLite": ["SQL"],
    "Snowflake": ["SQL", "Data warehousing"],
    "BigQuery": ["SQL", "Data warehousing"],
    "Redshift": ["SQL", "Data warehousing"],
    "Apache Spark": ["ETL"],
    "Airflow": ["ETL"],
    "dbt": ["SQL", "ETL"],
    "Tableau": ["Data visualization"],
    "Power BI": ["Data visualization"],
    "Looker": ["Data visualization"],
    "Kubernetes": ["Docker"],
    "Terraform": ["AWS"],
    "Jenkins": ["CI/CD"],
    "Selenium": ["Unit testing", "QA"],
    "Jira": ["Agile"],
}


def implied_by(skills: list[str]) -> dict[str, str]:
    """Skills implied by the given ones: {implied skill: the skill that implies it}."""
    out: dict[str, str] = {}
    for skill in skills:
        for implied in IMPLIES.get(skill, []):
            out.setdefault(implied, skill)
    return out


def skill_spans(text: str) -> list[tuple[int, int, str]]:
    """Every vocabulary-skill mention in ``text`` as (start, end, skill), sorted by position."""
    spans = []
    for skill, patterns in _compiled().items():
        for p in patterns:
            spans.extend((m.start(), m.end(), skill) for m in p.finditer(text))
    return sorted(spans)


def or_alternatives(text: str) -> list[tuple[str, str]]:
    """Pairs of skills joined by "or" / "/" in ``text``: "PyTorch or TensorFlow" -> [("PyTorch", "TensorFlow")].

    Only the skill right before and right after the "or" are paired, so in
    "Python and frameworks such as PyTorch or TensorFlow" Python is not an alternative.
    """
    spans = skill_spans(text)
    pairs = []
    for m in re.finditer(r"\s+or\s+|\s*/\s*", text, re.IGNORECASE):
        before = [s for s in spans if s[1] <= m.start()]
        after = [s for s in spans if s[0] >= m.end()]
        if before and after and before[-1][1] == m.start() and after[0][0] == m.end():
            pairs.append((before[-1][2], after[0][2]))
    return pairs
