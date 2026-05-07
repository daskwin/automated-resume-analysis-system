from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.services.model_downloader import ensure_all_models_exist
ensure_all_models_exist()

from app.api.routes import router
from app.db.database import init_db


app = FastAPI(
    title="Resume Screening Service",
    description="Service for automatic resume analysis and vacancy matching",
)


@app.on_event("startup")
def on_startup():
    init_db()


app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)

app.include_router(router)
