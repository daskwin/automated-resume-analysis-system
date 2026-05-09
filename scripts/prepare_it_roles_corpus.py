from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
AVISHEK_PATH = BASE_DIR / "data" / "raw" / "kaggle_avishekmajhi" / "Resume.csv"
UPDATED_PATH = BASE_DIR / "data" / "raw" / "kaggle_updated_resume" / "UpdatedResumeDataSet.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "resume_dataset_it_roles.csv"


ROLE_MAPPING = {
    "Database_Administrator": "Database Administrator",
    "Front_End_Developer": "Frontend Developer",
    "Java_Developer": "Java Developer",
    "Network_Administrator": "Network Administrator",
    "Project_manager": "Project Manager IT",
    "Python_Developer": "Python Developer",
    "Security_Analyst": "Security Engineer",
    "Software_Developer": "Software Developer",
    "Systems_Administrator": "Systems Administrator",
    "Web_Developer": "Web Developer",
    "Java Developer": "Java Developer",
    "Python Developer": "Python Developer",
    "DotNet Developer": "DotNet Developer",
    "SAP Developer": "SAP Developer",
    "Web Designing": "Web Developer",
    "Testing": "QA Engineer",
    "Automation Testing": "QA Engineer",
    "Data Science": "Data Scientist",
    "ETL Developer": "Data Engineer",
    "Hadoop": "Data Engineer",
    "Database": "Database Administrator",
    "DevOps Engineer": "DevOps Engineer",
    "Network Security Engineer": "Security Engineer",
    "Blockchain": "Blockchain Developer",
    "Business Analyst": "Business Analyst IT",
    "PMO": "Project Manager IT",
}

NON_IT_CATEGORIES = {
    "HR",
    "Sales",
    "Advocate",
    "Arts",
    "Mechanical Engineer",
    "Civil Engineer",
    "Electrical Engineering",
    "Health and fitness",
    "Operations Manager",
}


SYNTHETIC_BASE = {
    "QA Engineer": [
        "QA Engineer with experience in manual testing, regression testing, test case design, bug reporting, Postman, API testing, SQL validation, Jira, test documentation and release verification.",
        "Automation QA Engineer skilled in Selenium, Python, Pytest, REST API testing, UI testing, CI pipelines, automated regression suites and test reporting.",
        "Software Tester with experience in functional testing, integration testing, smoke testing, acceptance testing, defect tracking, Agile teams and quality assurance processes.",
        "QA Analyst experienced in writing test plans, preparing test scenarios, validating requirements, reporting defects and performing regression testing for web applications.",
        "Test Automation Engineer with Selenium WebDriver, Java, TestNG, REST Assured, API testing, Jenkins, Git and continuous testing experience.",
    ],
    "Data Engineer": [
        "Data Engineer with experience in Python, SQL, Apache Spark, PySpark, Airflow, Hadoop, Hive, HDFS, ETL pipelines, data marts and batch data processing.",
        "Big Data Engineer skilled in Spark, Kafka, Airflow, Hive, HDFS, Parquet, data lake architecture, workflow orchestration and production data pipelines.",
        "ETL Developer with experience designing SQL transformations, building data warehouse pipelines, processing raw data and creating clean datasets for analytics teams.",
        "Data Platform Engineer with hands-on experience in Spark SQL, Airflow DAGs, data quality checks, data ingestion, partitioned tables and pipeline monitoring.",
        "Analytics Engineer focused on ELT, SQL, Python, data modeling, dbt-like transformations, data marts, reporting layers and reliable analytical datasets.",
    ],
    "Data Scientist": [
        "Data Scientist with experience in Python, pandas, NumPy, scikit-learn, statistics, machine learning, classification, regression, clustering and model evaluation.",
        "Applied Data Scientist skilled in feature engineering, predictive modeling, A/B testing, NLP, XGBoost, model interpretation and business analytics.",
        "Machine Learning Data Scientist with experience in supervised learning, unsupervised learning, experiment design, SQL, Python and data-driven decision making.",
        "NLP Data Scientist experienced in text classification, embeddings, transformers, model validation, natural language processing and information extraction.",
        "Data Scientist focused on customer analytics, churn prediction, recommendation models, statistical analysis and presenting insights to stakeholders.",
    ],
    "DevOps Engineer": [
        "DevOps Engineer with experience in Docker, Kubernetes, Terraform, GitLab CI, Jenkins, Linux, monitoring, Prometheus, Grafana and CI/CD automation.",
        "Cloud DevOps Engineer skilled in AWS, Docker, Kubernetes, Helm, Terraform, infrastructure as code, deployment automation and production support.",
        "Site Reliability Engineer with experience in Kubernetes, Linux, observability, Prometheus, Grafana, incident response, alerting and service reliability.",
        "Platform Engineer responsible for CI/CD pipelines, container orchestration, environment automation, infrastructure monitoring and cloud resource management.",
        "Release Engineer with experience in build automation, Jenkins, GitHub Actions, Docker, Linux scripting, deployment pipelines and production releases.",
    ],
    "DotNet Developer": [
        ".NET Developer with experience in C#, ASP.NET Core, Entity Framework, SQL Server, REST APIs, microservices, Azure and backend application development.",
        "C# Backend Developer skilled in .NET, ASP.NET MVC, Web API, SQL Server, LINQ, Entity Framework, unit testing and enterprise software systems.",
        "DotNet Software Engineer with experience building web applications, REST services, database integrations, authentication, authorization and cloud deployment.",
        "Full stack .NET Developer with C#, ASP.NET Core, JavaScript, SQL Server, API development and production support experience.",
        ".NET Engineer focused on backend services, microservices architecture, message queues, SQL optimization, CI/CD and maintainable application design.",
    ],
    "SAP Developer": [
        "SAP Developer with experience in ABAP, SAP modules, reports, enhancements, BAPI, IDoc, SmartForms, SAP integration and business process automation.",
        "ABAP Developer skilled in SAP ERP, custom reports, user exits, BADIs, ALV, data migration, debugging and performance optimization.",
        "SAP Technical Consultant with experience in ABAP development, SAP interfaces, enhancements, forms, workflows and production support.",
        "SAP Engineer working with ABAP, functional specifications, technical documentation, module integration and enterprise system customization.",
        "SAP Developer experienced in business process analysis, custom development, data extraction, SAP reporting and integration with external systems.",
    ],
    "Business Analyst IT": [
        "IT Business Analyst with experience in requirements gathering, stakeholder interviews, BPMN, UML, user stories, acceptance criteria and functional specifications.",
        "System Analyst skilled in API documentation, SQL, UML diagrams, integration requirements, business process modeling and technical specifications.",
        "Business Analyst experienced in Agile projects, Jira, backlog management, software requirements, process analysis and communication with development teams.",
        "Functional Analyst with experience translating business needs into system requirements, preparing documentation, validating solutions and supporting implementation.",
        "IT Analyst working with stakeholders, developers and testers to define requirements, create user stories, describe business rules and manage project scope.",
    ],
    "Blockchain Developer": [
        "Blockchain Developer with experience in Solidity, smart contracts, Ethereum, Web3, decentralized applications, token standards and blockchain integrations.",
        "Web3 Developer skilled in smart contract development, Hardhat, Solidity, Ethereum, DeFi protocols, wallet integrations and decentralized applications.",
        "Blockchain Engineer with experience building distributed applications, smart contracts, crypto wallets, blockchain APIs and secure transaction workflows.",
        "Solidity Developer focused on Ethereum smart contracts, testing, auditing, gas optimization, Web3 libraries and decentralized finance applications.",
        "Blockchain Software Engineer with experience in smart contracts, Web3.js, Ethers.js, token contracts, decentralized systems and backend integrations.",
    ],
}


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    possible_text_cols = ["Resume", "resume", "Resume_str", "resume_text", "Text", "text"]
    possible_label_cols = ["Category", "category", "target_role", "Role", "role", "Label", "label"]

    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    label_col = next((col for col in possible_label_cols if col in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect columns. Found columns: {df.columns.tolist()}")

    return text_col, label_col


def load_and_normalize(path: Path, source: str) -> pd.DataFrame:
    print("=" * 100)
    print(f"Loading: {path}")

    df = pd.read_csv(path)

    print("Raw shape:", df.shape)
    print("Raw columns:", df.columns.tolist())

    text_col, label_col = detect_columns(df)

    df = df[[text_col, label_col]].copy()
    df = df.rename(columns={text_col: "resume_text", label_col: "original_category"})

    df["resume_text"] = df["resume_text"].apply(clean_text)
    df["original_category"] = df["original_category"].astype(str).str.strip()
    df["source"] = source

    df = df.dropna()
    df = df[df["resume_text"].str.len() > 100]

    print("Prepared shape:", df.shape)
    print("Original categories:")
    print(df["original_category"].value_counts())

    return df


def make_synthetic_variants(role: str, base_texts: list[str], needed: int) -> list[dict]:
    rows = []

    seniorities = [
        "Junior",
        "Middle",
        "Senior",
        "Lead",
        "Experienced",
        "Professional",
        "Skilled",
        "Results-driven",
    ]

    domains = [
        "fintech",
        "e-commerce",
        "enterprise software",
        "SaaS products",
        "banking systems",
        "analytics platforms",
        "cloud platforms",
        "internal business systems",
        "high-load web services",
        "data platforms",
        "telecom systems",
        "retail technology",
        "healthcare IT",
        "security products",
        "B2B platforms",
    ]

    responsibilities = [
        "collaborated with cross-functional teams",
        "prepared technical documentation",
        "participated in Agile development processes",
        "improved reliability and performance",
        "supported production systems",
        "implemented automated checks and monitoring",
        "worked closely with product and business stakeholders",
        "optimized existing workflows and services",
        "maintained clean and scalable solutions",
        "contributed to architecture discussions",
        "analyzed technical requirements",
        "supported release management",
        "created internal knowledge base articles",
        "reviewed existing implementation approaches",
        "participated in incident analysis",
    ]

    achievements = [
        "reduced processing time by 20 percent",
        "improved system stability",
        "increased automation coverage",
        "improved data quality and reporting accuracy",
        "reduced manual operational work",
        "improved monitoring and observability",
        "helped deliver several production releases",
        "improved team development workflow",
        "reduced number of production defects",
        "improved documentation quality",
        "optimized several critical business processes",
        "helped migrate legacy workflows",
        "improved service response time",
        "reduced support ticket volume",
        "standardized development practices",
    ]

    tools_context = [
        "Worked with Git, Linux, SQL and internal development tools.",
        "Used Jira, Confluence, Git and CI/CD tools in daily work.",
        "Worked with REST APIs, databases, logs and monitoring dashboards.",
        "Used Docker, Git, code reviews and automated testing practices.",
        "Collaborated with backend, frontend, analytics and infrastructure teams.",
        "Worked with production incidents, logs, metrics and troubleshooting.",
        "Created technical documentation, diagrams and implementation notes.",
        "Participated in sprint planning, backlog refinement and release preparation.",
        "Worked with business requirements, technical tasks and production support.",
        "Used modern engineering practices to improve maintainability and reliability.",
    ]

    idx = 0

    for base in base_texts:
        for seniority in seniorities:
            for domain in domains:
                for responsibility in responsibilities:
                    for achievement in achievements:
                        tools = tools_context[idx % len(tools_context)]

                        text = (
                            f"{seniority} {role}. "
                            f"{base} "
                            f"Worked in the {domain} domain and {responsibility}. "
                            f"Successfully {achievement}. "
                            f"{tools} "
                            f"Resume variant id: {role.replace(' ', '_')}_{idx}."
                        )

                        rows.append(
                            {
                                "resume_text": text,
                                "target_role": role,
                                "source": "synthetic_balancing_template",
                                "original_category": role,
                            }
                        )

                        idx += 1

                        if len(rows) >= needed:
                            return rows

    return rows


def add_synthetic_balancing(df: pd.DataFrame, min_count: int = 50) -> pd.DataFrame:
    rows = []

    counts = df["target_role"].value_counts().to_dict()

    for role, base_texts in SYNTHETIC_BASE.items():
        current_count = counts.get(role, 0)

        if current_count >= min_count:
            continue

        needed = min_count - current_count

        print(
            f"Balancing role '{role}': "
            f"current={current_count}, adding={needed}"
        )

        rows.extend(
            make_synthetic_variants(
                role=role,
                base_texts=base_texts,
                needed=needed,
            )
        )

    if not rows:
        return df

    synthetic_df = pd.DataFrame(rows)

    return pd.concat([df, synthetic_df], ignore_index=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    avishek_df = load_and_normalize(
        AVISHEK_PATH,
        source="kaggle_avishekmajhi_resume_dataset",
    )

    updated_df = load_and_normalize(
        UPDATED_PATH,
        source="kaggle_updated_resume_dataset",
    )

    raw_df = pd.concat([avishek_df, updated_df], ignore_index=True)

    print("\nCombined raw shape:", raw_df.shape)

    raw_df = raw_df[~raw_df["original_category"].isin(NON_IT_CATEGORIES)].copy()

    raw_df["target_role"] = raw_df["original_category"].map(ROLE_MAPPING)

    mapped_df = raw_df.dropna(subset=["target_role"]).copy()

    mapped_df = mapped_df[
        [
            "resume_text",
            "target_role",
            "source",
            "original_category",
        ]
    ]

    mapped_df = mapped_df.drop_duplicates(subset=["resume_text", "target_role"])

    print("\nBefore balancing:")
    print(mapped_df["target_role"].value_counts())

    mapped_df = add_synthetic_balancing(mapped_df, min_count=50)

    mapped_df = mapped_df.drop_duplicates(subset=["resume_text", "target_role"])
    mapped_df = mapped_df.sample(frac=1, random_state=42).reset_index(drop=True)

    mapped_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal shape:", mapped_df.shape)

    print("\nFinal target roles:")
    print(mapped_df["target_role"].value_counts())

    print("\nSources:")
    print(mapped_df["source"].value_counts())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
