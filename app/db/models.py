from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db.database import Base


class Vacancy(Base):
    """Database model for a job vacancy."""
    __tablename__ = "vacancies"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)

    description = Column(Text, nullable=False)
    requirements = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
