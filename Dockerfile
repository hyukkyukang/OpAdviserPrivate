# OpAdviser Docker Image
# Ubuntu 22.04 with MySQL 8.0, Python 3.10, and sysbench

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # MySQL Server
    mysql-server \
    mysql-client \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    # Sysbench for benchmarking
    sysbench \
    # Build tools (needed for some Python packages)
    build-essential \
    gcc \
    g++ \
    # Git and utilities
    git \
    curl \
    wget \
    vim \
    htop \
    # Required for psycopg2
    libpq-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements first (for Docker layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Install the autotune package in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/repo /var/run/mysqld /var/log/mysql

# Set MySQL permissions
RUN chown -R mysql:mysql /var/run/mysqld /var/log/mysql

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose MySQL port (optional, for external connections)
EXPOSE 3306

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command - start bash
CMD ["bash"]

