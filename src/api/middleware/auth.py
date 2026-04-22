import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/predict"):
            api_key = os.environ.get("API_KEY", "")
            auth = request.headers.get("Authorization", "")
            if not api_key or auth != f"Bearer {api_key}":
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)
