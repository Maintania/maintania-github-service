from app.db.session import engine
from app.db.base import Base

# Import models so SQLAlchemy registers them
from app.models import user, installation ,repository

def init_db():
    # Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
