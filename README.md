# Gani AI

A full-stack AI application with a Python FastAPI backend and a modern frontend interface.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Gani AI is a full-stack application that provides AI-powered functionality through a clean and intuitive interface. The project consists of:

- **Backend**: FastAPI-based REST API server
- **Frontend**: Modern web interface built with React/Next.js

## 📁 Project Structure

```
gani_ai/
├── src/
│   ├── backend/           # FastAPI backend application
│   │   ├── main.py        # Main application entry point
│   │   ├── requirements.txt
│   │   └── ...
│   └── frontend/          # Frontend application
│       ├── package.json
│       ├── app/
│       └── ...
├── README.md
└── ...
```

## 📋 Prerequisites

Before running this application, make sure you have the following installed:

- **Python 3.10**
- **Node.js 16+**
- **npm or yarn**
- **Conda** (recommended for Python environment management)
- **Git**

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Riandra4trend/gani_ai.git
cd gani_ai
```

### 2. Backend Setup

Navigate to the backend directory and set up the Python environment:

```bash
cd src/backend
```

Create a new conda environment:

```bash
conda create -n gani_ai python=3.9
conda activate gani_ai
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Frontend Setup

Navigate to the frontend directory:

```bash
cd ../frontend
```

Install Node.js dependencies:

```bash
npm install
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important**: 
- Replace `your_gemini_api_key_here` with your actual Gemini API key
- Never commit your `.env` file to version control
- Add `.env` to your `.gitignore` file

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and add it to your `.env` file

## 🏃‍♂️ Running the Application

### Start the Backend Server

1. Make sure your conda environment is activated:
```bash
conda activate gani_ai
```

2. Navigate to the backend directory:
```bash
cd src/backend
```

3. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The backend server will start on `http://localhost:8000`

### Start the Frontend Development Server

1. Open a new terminal window/tab
2. Navigate to the frontend directory:
```bash
cd src/frontend
```

3. Start the development server:
```bash
npm run dev
```

The frontend application will start on `http://localhost:3000` (or the next available port)

## 📖 API Documentation

Once the backend server is running, you can access:

- **Interactive API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative API Documentation**: `http://localhost:8000/redoc` (ReDoc)

## 🛠️ Development

### Backend Development

The backend is built with:
- **FastAPI**: Modern, fast web framework for building APIs
- **Python**: Core programming language
- **Uvicorn**: ASGI server for running the application

### Frontend Development

The frontend is built with modern web technologies. Check the `package.json` file for specific dependencies.

### Making Changes

1. Backend changes: The server will automatically reload thanks to the `--reload` flag
2. Frontend changes: The development server supports hot reloading

## 🚀 Deployment

### Production Backend

For production, you might want to use:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Production Frontend

Build the frontend for production:

```bash
npm run build
```

## 📝 Troubleshooting

### Common Issues

1. **Port already in use**: If you get port errors, either kill the process using the port or specify a different port
2. **Environment not activated**: Make sure to activate your conda environment before running the backend
3. **Missing dependencies**: Run `pip install -r requirements.txt` and `npm install` to ensure all dependencies are installed
4. **API key issues**: Verify your `GEMINI_API_KEY` is correctly set in the `.env` file

### Getting Help

If you encounter any issues:

1. Check the terminal output for error messages
2. Verify all prerequisites are installed
3. Ensure environment variables are properly set
4. Check that all dependencies are installed correctly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Backend Repository**: [https://github.com/Riandra4trend/gani_ai/tree/main/src/backend](https://github.com/Riandra4trend/gani_ai/tree/main/src/backend)
- **Frontend Repository**: [https://github.com/Riandra4trend/gani_ai/tree/main/src/frontend](https://github.com/Riandra4trend/gani_ai/tree/main/src/frontend)
- **Main Repository**: [https://github.com/Riandra4trend/gani_ai](https://github.com/Riandra4trend/gani_ai)

---

**Made with ❤️ by the Gani AI Team**