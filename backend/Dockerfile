# ---- DOCKERFILE ----
# Ye file Docker ko batati hai: "mera app kaise run karna hai"
# Teri machine pe bhi kaam karta hai, dost ki machine pe bhi, server pe bhi
# "Works on my machine" problem hamesha ke liye khatam

# Step 1: Base image - Python 3.11 slim (smaller = faster deployment)
FROM python:3.11-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy requirements FIRST (Docker caching trick)
# Agar requirements nahi badle toh pip install skip hoga = faster builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy rest of the code
COPY . .

# Step 5: Expose port (documentation purpose)
EXPOSE 8000

# Step 6: Start command - uvicorn is the production WSGI server
# NOT: python app.py (that's development only)
# --host 0.0.0.0 = accept connections from outside container
# --workers 2 = 2 parallel request handlers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]