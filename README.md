# Medical AI 🧬

Hey! 👋 Welcome to **Medical AI**.

This is a project I built to solve a real problem: understanding complex medical reports. We all get those lab results full of confusing numbers, right? I wanted to create an app where you can just upload a photo or PDF of your report, and it instantly tells you what's going on with your health.

It combines **Computer Vision (OCR)** to read the documents and **Machine Learning** to predict risks for things like Diabetes and Heart issues.

> **Live Demo:** https://medical-ai-fqnt.onrender.com/

---

## 💡 What Makes This Special?

I didn't just want a basic text reader. I built this to be **robust**:

* **It Reads "Messy" Files:** I spent a lot of time tuning the extractor. Whether it's a blurry phone photo, a rotated PDF, or a weirdly formatted CSV, this app digs through it to find the numbers that matter (Glucose, Cholesterol, etc.).
* **Smart Predictions:** It doesn't just show you the data; it analyzes it. I trained specific ML models to flag high risks for **Diabetes**, **Heart Disease**, and **Anemia**.
* **No More "Loading..." Stares:** I added a dynamic loading screen that tells you exactly what the AI is doing (Scanning -> Analyzing -> Reporting) so you aren't left wondering if it froze.
* **Glassmorphism UI:** Because medical apps don't have to look boring. It’s dark-themed, sleek, and works great on mobile.

---

## 🛠️ How It Works (The Tech)

* **The Brains:** Python & Flask
* **The Eyes (OCR):** Tesseract & OpenCV (Auto-detects if you're on Windows or Linux!)
* **The Logic:** Scikit-Learn & Pandas
* **The deploy:** Docker (Runs anywhere without dependency headaches)

---

## 🚀 How to Run It

I highly recommend using **Docker** because installing OCR tools manually can be a pain.

### Option 1: The Fast Way (Docker)
1.  Clone this repo:
    ```bash
    git clone [https://github.com/dhanush0254/medical-ai.git](https://github.com/dhanush0254/medical-ai.git)
    ```
2.  Build & Run:
    ```bash
    docker build -t medical-ai .
    docker run -p 5000:5000 medical-ai
    ```
3.  Go to `http://localhost:5000` and start uploading!

### Option 2: The Manual Way
If you want to run it directly on your machine, you'll need to install **Tesseract OCR** and **Poppler** first.
1.  Install the python libraries:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the server:
    ```bash
    python app.py
    ```

---

## 👥 The Team

Built with ❤️ by:
* **Dhanush**

---

## ⚠️ Just a Heads Up (Disclaimer)

**Please don't use this as a replacement for a real doctor.**
I built this as a demonstration of what AI can do. The predictions are virtual estimates based on data, but they aren't a medical diagnosis. Always check with a professional!
