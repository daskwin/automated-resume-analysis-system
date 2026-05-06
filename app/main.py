from fastapi import FastAPI


app = FastAPI(
    title="Resume Screening Service",
    description="Service for automatic resume analysis and vacancy matching",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "resumematch",
    }
