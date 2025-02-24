# Intelligent Tab Organizer

A Chrome extension to organize your tabs intelligently using machine learning.

## Setup

### Frontend
1. Navigate to the frontend directory
2. Run `npm install`
3. Run `npm start` for development or `npm run build` for production

### Backend
1. Navigate to the backend directory
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install flask flask-cors scikit-learn`
5. Run the Flask app: `python main.py`

## Loading the Extension
1. Build the frontend: `npm run build`
2. Open Chrome and go to `chrome://extensions/`
3. Enable "Developer mode"
4. Click "Load unpacked" and select the `build` folder in the frontend directory
