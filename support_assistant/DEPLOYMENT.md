# Local Deployment

The FastAPI app is served by `main:app` in the container on port 7860. The default mock mode requires no API key and supports deterministic policy and general-question examples.

Build from this directory with `docker build -t zepto-support .` and run with `docker run -p 7860:7860 zepto-support`.
