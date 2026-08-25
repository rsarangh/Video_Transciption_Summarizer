# Azure Deployment (App Service)

## 1) Push this repo to GitHub

If HTTPS auth fails, use a GitHub Personal Access Token (PAT):

```powershell
git -C c:/Users/rsara/Documents/py_srkv push -u origin main
```

When prompted:
- Username: your GitHub username
- Password: your GitHub PAT (not your GitHub password)

## 2) Create Azure Web App (Linux, Python)

In Azure Portal:
1. Create Resource -> Web App
2. Publish: Code
3. Runtime stack: Python 3.11
4. Operating System: Linux
5. Region: nearest to users

## 3) Connect GitHub deployment

In Web App -> Deployment Center:
1. Source: GitHub
2. Select repository: Video_Transciption_Summarizer
3. Branch: main
4. Save (Azure creates CI/CD)

## 4) Configure startup command

In Web App -> Configuration -> General settings:
- Startup Command:

```bash
gunicorn --bind=0.0.0.0:$PORT --timeout 180 app:app
```

## 5) Set environment variables

In Web App -> Configuration -> Application settings, add:
- OPENAI_API_KEY (optional if Gemini-only)
- GEMINI_API_KEY
- GEMINI_MODEL=gemini-3.6-flash
- AI_TIMEOUT_SECONDS=45
- MAX_TRANSCRIPT_CHARS=15000
- MAX_GEMINI_MODELS_TO_TRY=6

Save and restart the app.

## 6) Test after deploy

```bash
curl -X POST "https://<your-app-name>.azurewebsites.net/" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "video_id=https://youtu.be/OCdGyg0U6lI"
```

Health check:

```bash
curl "https://<your-app-name>.azurewebsites.net/health"
```

## 7) Android app integration

Use the Azure base URL in your app:
- POST https://<your-app-name>.azurewebsites.net/
- Body: application/x-www-form-urlencoded
- Field: video_id=<youtube_url>

Example JSON fields returned:
- summary
- transcription
- engine_used
- engine_errors (on failure)
