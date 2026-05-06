from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.resume_entity_extractor import extract_skills


MATCHING_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

SKILL_WEIGHT = 0.55
SEMANTIC_WEIGHT = 0.35
ROLE_WEIGHT = 0.10


class VacancyMatcher:
    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(MATCHING_EMBEDDING_MODEL_NAME)

    def match_resume_to_vacancy(
        self,
        resume_text: str,
        vacancy_text: str,
        resume_predicted_role: Optional[str] = None,
        vacancy_title: Optional[str] = None,
    ) -> dict[str, Any]:
        resume_skills = set(extract_skills(resume_text))
        vacancy_skills = set(extract_skills(vacancy_text))

        matched_skills = sorted(resume_skills & vacancy_skills)
        missing_skills = sorted(vacancy_skills - resume_skills)

        if vacancy_skills:
            skill_match_score = len(matched_skills) / len(vacancy_skills)
        else:
            skill_match_score = 0.0

        semantic_score = self.calculate_semantic_similarity(
            resume_text=resume_text,
            vacancy_text=vacancy_text,
        )

        role_match_score = self.calculate_role_match(
            resume_predicted_role=resume_predicted_role,
            vacancy_title=vacancy_title,
            vacancy_text=vacancy_text,
        )

        final_score = (
            SKILL_WEIGHT * skill_match_score
            + SEMANTIC_WEIGHT * semantic_score
            + ROLE_WEIGHT * role_match_score
        )

        return {
            "match_score": round(float(final_score), 4),
            "match_percent": round(float(final_score) * 100, 1),
            "skill_match_score": round(float(skill_match_score), 4),
            "semantic_score": round(float(semantic_score), 4),
            "role_match_score": round(float(role_match_score), 4),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "resume_skills": sorted(resume_skills),
            "vacancy_skills": sorted(vacancy_skills),
            "explanation": self.build_explanation(
                final_score=final_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                resume_predicted_role=resume_predicted_role,
                vacancy_title=vacancy_title,
            ),
        }

    def calculate_semantic_similarity(
        self,
        resume_text: str,
        vacancy_text: str,
    ) -> float:
        if not resume_text.strip() or not vacancy_text.strip():
            return 0.0

        embeddings = self.embedding_model.encode(
            [
                resume_text[:4000],
                vacancy_text[:4000],
            ],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarity = float(np.dot(embeddings[0], embeddings[1]))
        similarity = (similarity + 1) / 2

        return max(0.0, min(1.0, similarity))

    def calculate_role_match(
        self,
        resume_predicted_role: Optional[str],
        vacancy_title: Optional[str],
        vacancy_text: str,
    ) -> float:
        if not resume_predicted_role:
            return 0.0

        role_lower = resume_predicted_role.lower()
        vacancy_title_lower = (vacancy_title or "").lower()
        vacancy_text_lower = vacancy_text.lower()

        if role_lower in vacancy_title_lower:
            return 1.0

        if role_lower in vacancy_text_lower:
            return 0.8

        role_words = [
            word
            for word in role_lower.split()
            if len(word) > 2
        ]

        if role_words and all(word in vacancy_text_lower for word in role_words):
            return 0.7

        if any(word in vacancy_text_lower for word in role_words):
            return 0.4

        return 0.0

    def build_explanation(
        self,
        final_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
        resume_predicted_role: Optional[str],
        vacancy_title: Optional[str],
    ) -> str:
        if final_score >= 0.75:
            level = "The candidate is a strong match for the vacancy."
        elif final_score >= 0.50:
            level = "The candidate is a partial match for the vacancy."
        else:
            level = "The candidate is a weak match for the vacancy."

        vacancy_text = ""

        if vacancy_title:
            vacancy_text = f" Vacancy: {vacancy_title}."

        role_text = ""

        if resume_predicted_role:
            role_text = f" Candidate specialization: {resume_predicted_role}."

        matched_text = ""

        if matched_skills:
            matched_text = f" Matched skills: {', '.join(matched_skills[:8])}."

        missing_text = ""

        if missing_skills:
            missing_text = f" Missing skills: {', '.join(missing_skills[:8])}."

        return level + vacancy_text + role_text + matched_text + missing_text


vacancy_matcher = VacancyMatcher()
