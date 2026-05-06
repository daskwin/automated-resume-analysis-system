import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
NER_MODEL_DIR = BASE_DIR / "models" / "resume_bert_ner_model_chunked"


EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"
)

PHONE_PATTERN = re.compile(
    r"""
    (?<!\w)
    \+?\d[\d\s().\-–—]{7,}\d
    (?!\w)
    """,
    re.VERBOSE,
)

URL_PATTERN = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+|linkedin\.com/[^\s]+|github\.com/[^\s]+)",
    re.IGNORECASE,
)


SKILL_KEYWORDS = {
    # Programming languages
    "Python",
    "SQL",
    "Java",
    "JavaScript",
    "TypeScript",
    "C#",
    "C++",
    "C",
    "Go",
    "Golang",
    "Scala",
    "Kotlin",
    "Swift",
    "PHP",
    "Ruby",
    "R",
    "Bash",
    "Shell",
    "PowerShell",

    # Backend
    "FastAPI",
    "Django",
    "Flask",
    "Spring",
    "Spring Boot",
    "Node.js",
    "Express.js",
    "NestJS",
    "REST",
    "REST API",
    "GraphQL",
    "gRPC",
    "Microservices",
    "API Design",
    "OAuth",
    "JWT",

    # Frontend
    "React",
    "Vue",
    "Angular",
    "Redux",
    "Next.js",
    "Nuxt.js",
    "HTML",
    "CSS",
    "SASS",
    "SCSS",
    "Tailwind CSS",
    "Bootstrap",
    "Webpack",
    "Vite",

    # Data engineering
    "Apache Spark",
    "Spark",
    "PySpark",
    "Airflow",
    "Hadoop",
    "HDFS",
    "Hive",
    "Kafka",
    "Flink",
    "NiFi",
    "dbt",
    "ETL",
    "ELT",
    "Data Pipeline",
    "Data Pipelines",
    "Data Warehouse",
    "Data Lake",
    "Data Mart",
    "Data Modeling",
    "Data Quality",
    "Data Governance",
    "Parquet",
    "ORC",
    "Avro",
    "Iceberg",
    "Delta Lake",
    "Trino",
    "Presto",

    # Databases
    "PostgreSQL",
    "MySQL",
    "Oracle",
    "MS SQL",
    "SQL Server",
    "SQLite",
    "MongoDB",
    "Redis",
    "ClickHouse",
    "Greenplum",
    "Snowflake",
    "BigQuery",
    "Redshift",
    "Elasticsearch",
    "OpenSearch",
    "Cassandra",
    "DynamoDB",

    # Machine learning and AI
    "Machine Learning",
    "Deep Learning",
    "Data Science",
    "NLP",
    "Computer Vision",
    "Recommendation Systems",
    "Recommender Systems",
    "Time Series",
    "Feature Engineering",
    "Model Training",
    "Model Evaluation",
    "MLOps",
    "MLflow",
    "scikit-learn",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "PyTorch",
    "TensorFlow",
    "Keras",
    "pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Transformers",
    "Hugging Face",
    "BERT",
    "LLM",
    "RAG",

    # DevOps and infrastructure
    "Docker",
    "Docker Compose",
    "Kubernetes",
    "Terraform",
    "Ansible",
    "Helm",
    "Jenkins",
    "GitLab CI",
    "GitHub Actions",
    "CI/CD",
    "Linux",
    "Nginx",
    "Prometheus",
    "Grafana",
    "ELK",
    "AWS",
    "GCP",
    "Azure",
    "Yandex Cloud",
    "OpenShift",

    # Testing and QA
    "Selenium",
    "Pytest",
    "JUnit",
    "TestNG",
    "Postman",
    "Swagger",
    "API Testing",
    "Manual Testing",
    "Automation Testing",
    "Regression Testing",
    "Load Testing",
    "Performance Testing",
    "Unit Testing",
    "Integration Testing",
    "Test Cases",
    "Bug Reports",

    # Analytics and BI
    "Power BI",
    "Tableau",
    "Superset",
    "FineBI",
    "Excel",
    "Google Sheets",
    "Dashboard",
    "Dashboards",
    "Reporting",
    "A/B Testing",
    "Statistics",
    "Data Analysis",
    "Product Analytics",
    "Business Intelligence",
    "BI",
    "Metrics",
    "KPI",
    "Cohort Analysis",
    "Funnel Analysis",

    # Business analysis and product
    "Business Analysis",
    "System Analysis",
    "Requirements Analysis",
    "BPMN",
    "UML",
    "User Stories",
    "Acceptance Criteria",
    "Use Cases",
    "Jira",
    "Confluence",
    "Agile",
    "Scrum",
    "Kanban",
    "Product Management",
    "Roadmap",
    "Stakeholder Management",
    "Process Modeling",

    # Finance and accounting
    "Accounting",
    "Financial Analysis",
    "Financial Reporting",
    "Budgeting",
    "Forecasting",
    "Audit",
    "Tax",
    "IFRS",
    "GAAP",
    "Risk Management",
    "Credit Risk",
    "Market Risk",
    "Financial Modeling",
    "Cost Control",
    "Payroll",
    "Accounts Payable",
    "Accounts Receivable",

    # HR and recruiting
    "Recruitment",
    "HR",
    "Human Resources",
    "Talent Acquisition",
    "Onboarding",
    "Employee Relations",
    "HR Analytics",
    "Interviewing",
    "Sourcing",
    "Screening",
    "Performance Review",
    "Compensation",
    "Training",
    "Learning and Development",

    # Sales and customer service
    "Sales",
    "B2B Sales",
    "B2C Sales",
    "Lead Generation",
    "Negotiation",
    "CRM",
    "Salesforce",
    "HubSpot",
    "Customer Service",
    "Customer Support",
    "Client Relations",
    "Account Management",
    "Cold Calling",
    "Upselling",
    "Cross-selling",
    "Customer Success",

    # Marketing
    "Marketing",
    "Digital Marketing",
    "SEO",
    "SEM",
    "SMM",
    "Content Marketing",
    "Email Marketing",
    "Google Analytics",
    "Google Ads",
    "Meta Ads",
    "Brand Management",
    "Market Research",
    "Copywriting",
    "Campaign Management",
    "PR",
    "Public Relations",

    # Project and operations
    "Project Management",
    "Operations Management",
    "Process Improvement",
    "Lean",
    "Six Sigma",
    "Supply Chain",
    "Logistics",
    "Procurement",
    "Inventory Management",
    "Vendor Management",
    "Resource Planning",
    "Risk Assessment",
    "Quality Management",

    # Design
    "UI",
    "UX",
    "UI/UX",
    "Figma",
    "Adobe Photoshop",
    "Adobe Illustrator",
    "Adobe XD",
    "Design Systems",
    "Prototyping",
    "Wireframing",
    "User Research",
    "Usability Testing",
    "Graphic Design",

    # Fitness, sports and wellness
    "Fitness",
    "Personal Training",
    "Strength Training",
    "Functional Training",
    "Cardio Training",
    "Weight Training",
    "HIIT",
    "Pilates",
    "Yoga",
    "Stretching",
    "Mobility Training",
    "Group Training",
    "Sports Coaching",
    "Nutrition",
    "Meal Planning",
    "Weight Loss",
    "Body Composition",
    "Rehabilitation",
    "Injury Prevention",
    "Exercise Programming",
    "Client Assessment",
    "Fitness Assessment",
    "First Aid",
    "CPR",

    # Healthcare and medicine
    "Healthcare",
    "Patient Care",
    "Clinical Care",
    "Medical Records",
    "Electronic Health Records",
    "EHR",
    "EMR",
    "Nursing",
    "Medication Administration",
    "Vital Signs",
    "Patient Assessment",
    "Treatment Planning",
    "Care Planning",
    "Medical Terminology",
    "Health Education",
    "Public Health",
    "Emergency Care",
    "Triage",
    "Phlebotomy",
    "Laboratory Testing",
    "Diagnostics",
    "Infection Control",
    "HIPAA",
    "Clinical Documentation",
    "Case Management",

    # Psychology and social care
    "Counseling",
    "Mental Health",
    "Psychological Assessment",
    "Crisis Intervention",
    "Social Work",
    "Behavioral Therapy",
    "CBT",
    "Group Therapy",
    "Family Therapy",
    "Client Support",
    "Care Coordination",

    # Education
    "Teaching",
    "Curriculum Development",
    "Lesson Planning",
    "Classroom Management",
    "Student Assessment",
    "Tutoring",
    "E-learning",
    "Instructional Design",
    "Educational Technology",
    "Special Education",
    "Academic Advising",

    # Hospitality, food and service
    "Hospitality",
    "Food Safety",
    "Menu Planning",
    "Cooking",
    "Baking",
    "Catering",
    "Restaurant Management",
    "Kitchen Management",
    "Inventory Control",
    "HACCP",
    "Customer Experience",
    "Event Planning",
    "Front Desk",
    "Housekeeping",
    "Reservation Management",

    # Legal and administrative
    "Legal Research",
    "Legal Writing",
    "Contract Review",
    "Compliance",
    "Document Management",
    "Administrative Support",
    "Office Management",
    "Scheduling",
    "Data Entry",
    "Record Keeping",
    "Microsoft Office",

    # General soft skills
    "Communication",
    "Leadership",
    "Teamwork",
    "Problem Solving",
    "Critical Thinking",
    "Analytical Thinking",
    "Presentation Skills",
    "Time Management",
    "Adaptability",
    "Creativity",
    "Decision Making",
    "Collaboration",
    "Mentoring",
    "Conflict Resolution",
    "Empathy",
    "Active Listening",
    "Stress Management",
    "Attention to Detail",
    "Multitasking",
    "English",
    "German",
    "French",
    "Spanish",
}


UNIVERSITY_WORDS = {
    "university",
    "college",
    "institute",
    "school",
    "academy",
}

COMPANY_WORDS = {
    "company",
    "technologies",
    "technology",
    "solutions",
    "systems",
    "analytics",
    "bank",
    "group",
    "ltd",
    "llc",
    "inc",
    "corp",
    "corporation",
}


def unique_keep_order(values: list[str]) -> list[str]:
    """Remove duplicates while keeping the original order."""
    seen = set()
    result = []

    for value in values:
        value = str(value).strip(" ,.;:\n\t")

        if not value:
            continue

        key = value.lower()

        if key not in seen:
            seen.add(key)
            result.append(value)

    return result


def normalize_bert_text(value: str) -> str:
    """Clean tokenization artifacts from BERT output."""
    value = value.replace(" ##", "")
    value = value.replace("##", "")
    value = value.replace(" ' ", "'")
    value = value.replace(" ,", ",")
    value = value.replace(" .", ".")
    value = value.strip(" ,.;:\n\t")

    return value


def normalize_phone_candidate(value: str) -> str:
    """Normalize spaces and dashes in a phone candidate."""
    value = str(value)

    value = value.replace("\u00a0", " ")
    value = value.replace("–", "-")
    value = value.replace("—", "-")

    value = re.sub(r"\s+", " ", value)
    value = value.strip(" ,.;:\n\t")

    return value


def is_valid_phone(value: str) -> bool:
    """Filter out dates and invalid phone-like fragments."""
    value = normalize_phone_candidate(value)

    if not value:
        return False

    digits = re.sub(r"\D", "", value)
    digits_count = len(digits)

    if digits_count < 10 or digits_count > 15:
        return False

    if re.fullmatch(r"\d{4}\s*[-–—]\s*\d{4}", value):
        return False

    if re.search(r"\b(?:19|20)\d{2}\b", value):
        return False

    has_phone_separator = any(
        char in value
        for char in ["+", " ", "-", ".", "(", ")"]
    )

    if not has_phone_separator:
        return False

    return True


def extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    return unique_keep_order(EMAIL_PATTERN.findall(text))


def extract_phones(text: str) -> list[str]:
    """Extract phone numbers from text."""
    result = []
    text = text.replace("\u00a0", " ")

    for match in PHONE_PATTERN.finditer(text):
        phone = normalize_phone_candidate(match.group(0))

        if is_valid_phone(phone):
            result.append(phone)

    return unique_keep_order(result)


def extract_links(text: str) -> list[str]:
    """Extract common profile and web links."""
    return unique_keep_order(URL_PATTERN.findall(text))


def extract_skills(text: str) -> list[str]:
    """Extract skills using a predefined keyword dictionary."""
    text_lower = text.lower()
    matched = []

    for skill in SKILL_KEYWORDS:
        skill_lower = skill.lower()
        pattern = (
            r"(?<![a-zA-Z0-9+#.])"
            + re.escape(skill_lower)
            + r"(?![a-zA-Z0-9+#.])"
        )

        if re.search(pattern, text_lower):
            matched.append(skill)

    return sorted(unique_keep_order(matched), key=lambda x: x.lower())


def looks_like_education(value: str) -> bool:
    """Check whether a value looks like an education entity."""
    value_lower = value.lower()

    return any(word in value_lower for word in UNIVERSITY_WORDS)


def looks_like_company(value: str) -> bool:
    """Check whether a value looks like a company name."""
    value_lower = value.lower()

    return any(word in value_lower for word in COMPANY_WORDS)


def clean_company_name(value: str) -> str:
    """Clean noisy company name fragments."""
    value = str(value).strip(" ,.;:\n\t")

    stop_words = [
        "Email",
        "Phone",
        "Location",
        "LinkedIn",
        "GitHub",
        "Summary",
        "Work Experience",
        "Education",
        "Skills",
    ]

    for stop_word in stop_words:
        pattern = rf"\s+{re.escape(stop_word)}\b.*$"
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

    return value.strip(" ,.;:\n\t")


def deduplicate_companies(values: list[str]) -> list[str]:
    """Clean and deduplicate company names."""
    cleaned = []

    for value in values:
        company = clean_company_name(value)

        if company and len(company.split()) <= 6:
            cleaned.append(company)

    return unique_keep_order(cleaned)


def extract_company_candidates_by_rules(text: str) -> list[str]:
    """Extract additional company candidates using simple rules."""
    companies = []

    company_keywords = [
        "Company",
        "Technologies",
        "Technology",
        "Solutions",
        "Systems",
        "Analytics",
        "Bank",
        "Group",
        "Ltd",
        "LLC",
        "Inc",
        "Corp",
        "Corporation",
    ]

    for line in text.splitlines():
        line = line.strip(" ,.;:\n\t")

        if not line or len(line.split()) > 8:
            continue

        if any(keyword.lower() in line.lower() for keyword in company_keywords):
            companies.append(line)

    return unique_keep_order(companies)


class ResumeEntityExtractor:

    def __init__(self) -> None:
        if not NER_MODEL_DIR.exists():
            raise FileNotFoundError(f"NER model not found: {NER_MODEL_DIR}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
        self.model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    def extract_bert_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract entities with a token classification model."""
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()

        encoded.pop("token_type_ids", None)

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predictions = torch.argmax(probabilities, dim=-1).tolist()
        scores = torch.max(probabilities, dim=-1).values.tolist()

        raw_entities = []
        current_entity = None

        for token_idx, label_id in enumerate(predictions):
            start, end = offsets[token_idx]

            if start == end:
                continue

            label = self.id2label[int(label_id)]
            score = float(scores[token_idx])

            if label == "O":
                if current_entity is not None:
                    raw_entities.append(current_entity)
                    current_entity = None
                continue

            if "-" not in label:
                continue

            prefix, entity_type = label.split("-", 1)

            if (
                prefix == "B"
                or current_entity is None
                or current_entity["entity_group"] != entity_type
            ):
                if current_entity is not None:
                    raw_entities.append(current_entity)

                current_entity = {
                    "entity_group": entity_type,
                    "start": start,
                    "end": end,
                    "scores": [score],
                }
            else:
                current_entity["end"] = end
                current_entity["scores"].append(score)

        if current_entity is not None:
            raw_entities.append(current_entity)

        result = []

        for entity in raw_entities:
            value = text[entity["start"]:entity["end"]]
            value = normalize_bert_text(value)

            if not value:
                continue

            result.append(
                {
                    "word": value,
                    "entity_group": entity["entity_group"],
                    "score": sum(entity["scores"]) / len(entity["scores"]),
                }
            )

        return result

    def extract(self, text: str) -> dict[str, Any]:
        """Extract all supported resume entities."""
        if not text or not text.strip():
            return self.empty_result()

        ner_text = text[:3000]
        ner_entities = self.extract_bert_entities(ner_text)

        person_names = []
        designations = []
        companies = []
        locations = []
        education = []

        for entity in ner_entities:
            value = normalize_bert_text(entity.get("word", ""))
            label = entity.get("entity_group", "")
            score = float(entity.get("score", 0.0))

            if not value or score < 0.50:
                continue

            if label == "PERSON":
                if looks_like_education(value):
                    education.append(value)
                elif looks_like_company(value):
                    companies.append(value)
                else:
                    person_names.append(value)

            elif label == "DESIGNATION":
                designations.append(value)

            elif label == "COMPANY":
                companies.append(value)

            elif label == "LOCATION":
                locations.append(value)

            elif label == "EDUCATION":
                education.append(value)

        companies.extend(extract_company_candidates_by_rules(text))

        return {
            "emails": extract_emails(text),
            "phones": extract_phones(text),
            "links": extract_links(text),
            "skills": extract_skills(text),
            "person_names": unique_keep_order(person_names),
            "designations": unique_keep_order(designations),
            "companies": deduplicate_companies(companies),
            "locations": unique_keep_order(locations),
            "education": unique_keep_order(education),
        }

    @staticmethod
    def empty_result() -> dict[str, list[str]]:
        """Return an empty entity response."""
        return {
            "emails": [],
            "phones": [],
            "links": [],
            "skills": [],
            "person_names": [],
            "designations": [],
            "companies": [],
            "locations": [],
            "education": [],
        }


resume_entity_extractor = ResumeEntityExtractor()
