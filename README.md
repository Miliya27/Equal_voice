# EqualVoice

EqualVoice is a web-based sign language support platform built with React, FastAPI, MediaPipe, and PyTorch. It combines live sign interpretation, guided learning, a video manual, and a simple text-to-sign animation view in one project.

## Overview

The project is split into two parts:

- `frontend`: a React + Vite web app in the repository root
- `backend`: a FastAPI inference service inside `backend/`

The frontend handles the user interface, webcam access, and speech output. The backend receives webcam frames, extracts hand landmarks with MediaPipe, runs a trained PyTorch model, and returns the predicted sign label.

## Features

- Live sign-to-text interpretation from webcam input
- Speech output for predicted signs
- Practice-based sign learning mode
- Video-based sign manual
- Text-to-sign stickman animation for a small supported word set
- Simple dark mode toggle

## Tech Stack

- Frontend: React, Vite, Tailwind CSS
- Backend: FastAPI, Uvicorn
- ML / CV: PyTorch, MediaPipe, OpenCV
- Browser APIs: Webcam access, Web Speech API

## Project Structure

```text
Equal_voice/
|-- backend/
|   |-- app/
|   |   |-- api_server.py
|   |   |-- predictor.py
|   |   `-- model_loader.py
|   |-- ml/
|   |   |-- data/
|   |   `-- src/
|   |-- requirements.txt
|   |-- runtime.txt
|   `-- Dockerfile
|-- public/
|   `-- videos/
|-- src/
|   |-- components/
|   |-- pages/
|   |-- App.jsx
|   `-- main.jsx
|-- package.json
`-- README.md
```

## Main Pages

- `Landing`: project introduction and entry points
- `Interpreter`: webcam-based sign prediction using the backend API
- `Learning`: watch sign videos and practice signs with webcam input
- `Manual`: browse sign videos
- `Text -> Sign`: play stickman animations for supported words
- `QR`: generate a QR code for the app
- `About`: summary of the project and stack

## Prerequisites

Before running the project, make sure you have:

- Node.js 18+ installed
- npm installed
- Python 3.10 installed

Important:

- The backend is configured for Python `3.10.x`
- `mediapipe==0.10.32` is typically the package most likely to fail on newer Python versions such as `3.13`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Miliya27/Equal_voice.git
cd Equal_voice
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Set up the backend virtual environment

On Windows PowerShell:

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `py -3.10` is not available, install Python 3.10 first and retry.

## Running the Project

You need two terminals: one for the frontend and one for the backend.

### Terminal 1: Start the frontend

From the project root:

```bash
npm run dev
```

Vite will usually start on:

```text
http://localhost:5173
```

### Terminal 2: Start the backend

From `backend/` with the virtual environment activated:

```bash
uvicorn app.api_server:app --reload --host 127.0.0.1 --port 8000
```

The API will run on:

```text
http://127.0.0.1:8000
```

## How It Works

1. The browser captures webcam frames.
2. The frontend sends the image to `POST /predict`.
3. The backend decodes the image with OpenCV.
4. MediaPipe extracts hand landmarks.
5. The model predicts the most likely sign label.
6. The frontend shows the result and can speak it aloud.

## API

### `GET /`

Health check endpoint.

Response:

```json
{ "status": "API running" }
```

### `POST /predict`

Accepts a single uploaded image file and returns the predicted sign.

Response example:

```json
{ "prediction": "HELLO" }
```

## Supported Modes

### Fully frontend-only

These pages do not need the backend API:

- Landing
- Manual
- About
- Most of Text -> Sign

### Require backend

These features require the FastAPI server to be running:

- Live Interpreter
- Learning practice mode

## Known Limitations

- The frontend hardcodes the backend URL as `http://127.0.0.1:8000`
- The app uses internal React state for navigation instead of React Router
- The QR page builds a `/manual` URL, but the app does not currently define route-based navigation
- `Learning.jsx` references `up.mp4` and `down.mp4`, but those files are not present in `public/videos`
- Text-to-sign animations only work for the words that have matching JSON files in `src/components/stickman/`

## Troubleshooting

### Backend install fails

Cause:

- Usually a Python version mismatch

Fix:

- Use Python 3.10
- Recreate the virtual environment
- Reinstall with `pip install -r requirements.txt`

### Camera does not work

Check:

- Browser camera permission is allowed
- Another app is not blocking webcam access
- You opened the Vite frontend, not the HTML file directly

### Predictions are not showing

Check:

- Backend is running on `127.0.0.1:8000`
- Frontend is running on Vite
- Browser console and backend terminal for request errors

## Development Scripts

From the project root:

```bash
npm run dev
npm run build
npm run preview
npm run lint
```

## Future Improvements

- Add proper route-based navigation
- Move backend URL to environment configuration
- Expand the trained sign vocabulary
- Improve mobile responsiveness
- Add missing video assets referenced by the learning page
- Add setup automation and clearer health checks

## License

No license file is currently included in this repository.
