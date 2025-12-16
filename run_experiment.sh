#!/bin/bash
# OpAdviser Quick Experiment Script
# This script prepares the database and runs the 10-minute OpAdviser experiment

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   OpAdviser Quick Experiment${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
DB_NAME="sbrw"
DB_USER="root"
DB_PASS="password"
TABLES=10
TABLE_SIZE=10000
THREADS=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tables)
            TABLES="$2"
            shift 2
            ;;
        --table-size)
            TABLE_SIZE="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --full)
            # Full experiment settings
            TABLES=50
            TABLE_SIZE=100000
            THREADS=80
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tables N      Number of tables (default: 10)"
            echo "  --table-size N  Rows per table (default: 10000)"
            echo "  --threads N     Benchmark threads (default: 8)"
            echo "  --full          Use full experiment settings (50 tables, 100k rows, 80 threads)"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuration:${NC}"
echo "  Database: $DB_NAME"
echo "  Tables: $TABLES"
echo "  Table Size: $TABLE_SIZE rows"
echo "  Threads: $THREADS"
echo ""

# Step 1: Create database
echo -e "${YELLOW}Step 1: Creating database...${NC}"
mysql -u$DB_USER -p$DB_PASS -e "DROP DATABASE IF EXISTS $DB_NAME; CREATE DATABASE $DB_NAME;"
echo -e "${GREEN}Database '$DB_NAME' created!${NC}"

# Step 2: Prepare sysbench data
echo -e "${YELLOW}Step 2: Preparing sysbench data (this may take a few minutes)...${NC}"
sysbench \
    --db-driver=mysql \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=$DB_USER \
    --mysql-password=$DB_PASS \
    --mysql-db=$DB_NAME \
    --tables=$TABLES \
    --table_size=$TABLE_SIZE \
    --threads=$THREADS \
    oltp_read_write prepare

echo -e "${GREEN}Sysbench data prepared!${NC}"

# Step 3: Run a quick benchmark test
echo -e "${YELLOW}Step 3: Running quick benchmark test (10 seconds)...${NC}"
sysbench \
    --db-driver=mysql \
    --mysql-host=localhost \
    --mysql-port=3306 \
    --mysql-user=$DB_USER \
    --mysql-password=$DB_PASS \
    --mysql-db=$DB_NAME \
    --tables=$TABLES \
    --table_size=$TABLE_SIZE \
    --threads=$THREADS \
    --time=10 \
    --report-interval=5 \
    oltp_read_write run

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "You can now run OpAdviser experiments:"
echo ""
echo -e "${YELLOW}# 10-minute quick experiment:${NC}"
echo "python scripts/optimize.py --config=scripts/config_cpu_io_opadviser_10min.ini"
echo ""
echo -e "${YELLOW}# Full CPU/IO focused experiment:${NC}"
echo "python scripts/optimize.py --config=scripts/config_cpu_io_opadviser.ini"
echo ""

