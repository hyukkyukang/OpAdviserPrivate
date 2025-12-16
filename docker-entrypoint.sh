#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   OpAdviser Docker Container${NC}"
echo -e "${GREEN}========================================${NC}"

# Start MySQL if not already running
if ! pgrep -x "mysqld" > /dev/null; then
    echo -e "${YELLOW}Starting MySQL server...${NC}"
    
    # Initialize MySQL data directory if needed
    if [ ! -d "/var/lib/mysql/mysql" ]; then
        echo -e "${YELLOW}Initializing MySQL data directory...${NC}"
        mysqld --initialize-insecure --user=mysql
    fi
    
    # Start MySQL in the background
    mysqld --user=mysql &
    
    # Wait for MySQL to be ready
    echo -e "${YELLOW}Waiting for MySQL to be ready...${NC}"
    for i in {1..30}; do
        if mysqladmin ping -h localhost --silent 2>/dev/null; then
            echo -e "${GREEN}MySQL is ready!${NC}"
            break
        fi
        sleep 1
    done
    
    # Set root password if this is first run
    if mysql -uroot -e "SELECT 1" 2>/dev/null; then
        echo -e "${YELLOW}Setting MySQL root password...${NC}"
        mysql -uroot -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
        mysql -uroot -ppassword -e "FLUSH PRIVILEGES;"
        echo -e "${GREEN}Root password set to 'password'${NC}"
    fi
else
    echo -e "${GREEN}MySQL is already running${NC}"
fi

# Verify MySQL connection
if mysql -uroot -ppassword -e "SELECT VERSION();" > /dev/null 2>&1; then
    echo -e "${GREEN}MySQL connection verified!${NC}"
    mysql -uroot -ppassword -e "SELECT VERSION() AS 'MySQL Version';"
else
    echo -e "${YELLOW}Warning: Could not verify MySQL connection${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Quick Start Commands${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "# Prepare database and run benchmark:"
echo "  ./run_experiment.sh"
echo ""
echo "# Or manually:"
echo "  mysql -uroot -ppassword -e 'CREATE DATABASE IF NOT EXISTS sbrw;'"
echo "  sysbench oltp_read_write --db-driver=mysql --mysql-user=root \\"
echo "    --mysql-password=password --mysql-db=sbrw --tables=10 \\"
echo "    --table_size=10000 prepare"
echo ""
echo "# Run OpAdviser (10-minute mode):"
echo "  python scripts/optimize.py --config=scripts/config_cpu_io_opadviser_10min.ini"
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""

# Execute the command passed to docker run
exec "$@"

