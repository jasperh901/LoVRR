import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lovrr.inference.pipeline import LoVRR


class InferenceRequest(BaseModel):
    video_path: str
    question: str
    problem_type: str | None = None
    max_frames: int | None = Field(None, ge=2)
    max_new_tokens: int | None = Field(None, ge=1)


def create_app(config, runtime=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime if runtime is not None else LoVRR.from_config(config)
        yield
        app.state.runtime = None

    app = FastAPI(title="LoVRR", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        loaded = getattr(app.state, "runtime", None) is not None
        return {"status": "healthy" if loaded else "initializing", "model_loaded": loaded}

    @app.post("/answer")
    @app.post("/infer")
    def answer(request: InferenceRequest):
        instance = getattr(app.state, "runtime", None)
        if instance is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        try:
            return instance.answer(**request.model_dump())
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser(description="LoVRR video reasoning service")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    logging.basicConfig(level=logging.INFO)
    server = config.get("server", {})
    uvicorn.run(
        create_app(config),
        host=args.host or server.get("host", "127.0.0.1"),
        port=args.port if args.port is not None else server.get("port", 8000),
    )


if __name__ == "__main__":
    main()
