# Start from this image
FROM python:3.11.4-slim

# Set the working directory in the container
WORKDIR /workspace
# COPY . /workspace

# Install Git, gcc, g++, cmake 
RUN apt-get update && \
    apt-get install -y \
    git \
    gcc \
    g++ \
    cmake \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Clone the GitHub repository
RUN git clone https://github.com/cantaro86/Financial-Models-Numerical-Methods.git .

# Install requirments
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --requirement requirements.txt

# Old style jupyter notebooks
RUN pip install nbclassic

# Install local package
RUN pip install .

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook server
CMD ["jupyter", "nbclassic", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]


#################################################################

# 1. BUILD
# docker build -t fmnm .    

# 2. CREATE CONTAINER  
# docker run --rm -d -p 8888:8888 --name Numeric_Finance fmnm

# 3. OPEN IN BROWSER 
# http://localhost:8888/lab
# or
# http://localhost:8888/lab

# OR 

# 1. docker-compose up --build -d --remove-orphans
# 2. docker-compose down --rmi all --volumes --remove-orphans
