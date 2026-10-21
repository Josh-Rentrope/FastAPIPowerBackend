# The API image. Built by App Platform from the GitHub source, or locally and
# pushed to a registry -- whichever `api_from_image` selects in the Terraform.
#
# python:3.12-slim and psycopg2-binary together are what keep this small: the
# binary wheel bundles libpq, so no postgresql-dev, no gcc, and no build step
# beyond pip. App Platform gives a build 8 GB of RAM and disk and will not raise
# it, which is why the ETL dependencies (pyshp, pycountry) are NOT installed
# here -- they are needed to build the database, not to serve it.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first, so a source-only change does not reinstall dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sessions.py FastAPIPostgresServer.py ./
COPY api/ ./api/
# The eligibility criteria and evaluator. Missing this is a crash at import, not a
# degraded feature -- the API will not start without the module.
COPY qualification/ ./qualification/

# Run as a non-root user. Nothing here writes to the filesystem, so there is no
# reason for the process to be able to.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# App Platform injects PORT; 8080 is the default its health checks expect.
ENV PORT=8080
EXPOSE 8080

# No --reload and no workers>1: the container is replaced on deploy rather than
# restarted, and a single process keeps the connection pool predictable.
CMD ["sh", "-c", "uvicorn FastAPIPostgresServer:app --host 0.0.0.0 --port ${PORT}"]
