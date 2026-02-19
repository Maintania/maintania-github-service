from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, RedirectResponse

router = APIRouter()

router.get("/github/callback")
async def github_callback(request: Request):
   print("FULL URL:", request.url)
   print("QUERY:", request.query_params)

   installation_id = request.query_params.get("installation_id")

   if not installation_id:
        return JSONResponse({"error": "missing installation_id"}, status_code=400)

   print("Installation ID:", installation_id)

   installation_id = request.query_params.get("installation_id")

   
   print("Installation ID:", installation_id)

   INSTALLATIONS["current"] = installation_id

   return RedirectResponse("http://localhost:3000/dashboard") 