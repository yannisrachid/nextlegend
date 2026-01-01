from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from settings import settings

engine = create_engine(
    settings.database_url,
    future=True,
    connect_args={"prepare_threshold": 0},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
