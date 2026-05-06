from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Vacancy
from app.services.embedding_classifier import resume_embedding_classifier
from app.services.it_role_classifier import it_role_classifier
from app.services.resume_entity_extractor import resume_entity_extractor
from app.services.text_extractor import (
    EmptyTextError,
    UnsupportedFileTypeError,
    extract_text_from_file,
)
from app.services.vacancy_matcher import vacancy_matcher


router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def predict_general_domain(text: str) -> dict[str, Any]:
    return resume_embedding_classifier.predict(
        text=text,
        top_k=3,
    )


def predict_it_role(text: str) -> dict[str, Any]:
    return it_role_classifier.predict(
        text=text,
        top_k=3,
    )


def build_analysis_response(
    filename: str,
    extracted_text: str,
    general_prediction: dict[str, Any],
    it_role_prediction: dict[str, Any],
    entities: dict[str, Any],
    vacancy_matches: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    predicted_domain = general_prediction["predicted_category"]
    top_domains = general_prediction["top_3_categories"]

    return {
        "filename": filename,
        "text_length": len(extracted_text),
        "text_preview": extracted_text,
        "predicted_domain": predicted_domain,
        "top_3_domains": top_domains,
        "confidence": general_prediction["confidence"],
        "model_type": general_prediction.get("model_type"),
        "embedding_model": general_prediction.get("embedding_model"),
        "classifier_name": general_prediction.get("classifier_name"),
        "it_role": {
            "predicted_role": it_role_prediction["predicted_role"],
            "confidence": it_role_prediction["confidence"],
            "top_roles": it_role_prediction["top_roles"],
            "model_type": it_role_prediction["model_type"],
            "embedding_model": it_role_prediction.get("embedding_model"),
            "classifier_name": it_role_prediction.get("classifier_name"),
        },
        "entities": entities,
        "vacancy_matches": vacancy_matches or [],
        "status": "success",
    }


def vacancy_to_dict(vacancy: Vacancy) -> dict[str, Any]:
    return {
        "id": vacancy.id,
        "title": vacancy.title,
        "company": vacancy.company,
        "location": vacancy.location,
        "description": vacancy.description,
        "requirements": vacancy.requirements,
        "created_at": vacancy.created_at.isoformat() if vacancy.created_at else None,
    }


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "resume-analysis",
        "models": {
            "general_classifier": "enabled",
            "it_role_classifier": "enabled",
            "entity_extractor": "enabled",
            "vacancy_matching": "enabled",
        },
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@router.post("/api/vacancies")
def create_vacancy(
    title: str = Form(...),
    company: str = Form(""),
    location: str = Form(""),
    description: str = Form(...),
    requirements: str = Form(""),
    db: Session = Depends(get_db),
):
    vacancy = Vacancy(
        title=title,
        company=company,
        location=location,
        description=description,
        requirements=requirements,
    )

    db.add(vacancy)
    db.commit()
    db.refresh(vacancy)

    return {
        "status": "success",
        "vacancy": vacancy_to_dict(vacancy),
    }


@router.get("/api/vacancies")
def list_vacancies(db: Session = Depends(get_db)):
    vacancies = db.query(Vacancy).order_by(Vacancy.created_at.desc()).all()

    return {
        "vacancies": [
            vacancy_to_dict(vacancy)
            for vacancy in vacancies
        ],
    }


@router.delete("/api/vacancies/{vacancy_id}")
def delete_vacancy(
    vacancy_id: int,
    db: Session = Depends(get_db),
):
    vacancy = db.query(Vacancy).filter(Vacancy.id == vacancy_id).first()

    if vacancy is None:
        raise HTTPException(status_code=404, detail="Vacancy not found")

    db.delete(vacancy)
    db.commit()

    return {
        "status": "success",
        "deleted_vacancy_id": vacancy_id,
    }


@router.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    match_vacancies: bool = Form(False),
    db: Session = Depends(get_db),
):
    filename = file.filename or "uploaded_resume"
    content = await file.read()

    try:
        extracted_text = extract_text_from_file(
            filename=filename,
            content=content,
        )
    except UnsupportedFileTypeError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except EmptyTextError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from file '{filename}': {str(error)}",
        )

    try:
        general_prediction = predict_general_domain(extracted_text)
        it_role_prediction = predict_it_role(extracted_text)
        entities = resume_entity_extractor.extract(extracted_text)

        vacancy_matches = []

        if match_vacancies:
            vacancies = db.query(Vacancy).order_by(Vacancy.created_at.desc()).all()

            for vacancy in vacancies:
                vacancy_text = "\n".join(
                    [
                        vacancy.title or "",
                        vacancy.company or "",
                        vacancy.location or "",
                        vacancy.description or "",
                        vacancy.requirements or "",
                    ]
                )

                match_result = vacancy_matcher.match_resume_to_vacancy(
                    resume_text=extracted_text,
                    vacancy_text=vacancy_text,
                    resume_predicted_role=it_role_prediction["predicted_role"],
                    vacancy_title=vacancy.title,
                )

                vacancy_matches.append(
                    {
                        "vacancy_id": vacancy.id,
                        "title": vacancy.title,
                        "company": vacancy.company,
                        "location": vacancy.location,
                        **match_result,
                    }
                )

            vacancy_matches = sorted(
                vacancy_matches,
                key=lambda item: item["match_score"],
                reverse=True,
            )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze resume '{filename}': {str(error)}",
        )

    return build_analysis_response(
        filename=filename,
        extracted_text=extracted_text,
        general_prediction=general_prediction,
        it_role_prediction=it_role_prediction,
        entities=entities,
        vacancy_matches=vacancy_matches,
    )
