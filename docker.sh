#!/bin/bash
echo "# Generated by NetworkManager
search cluster. cluster
nameserver 141.223.1.2
nameserver 1.0.0.1" | sudo tee /etc/resolv.conf
mkdir -p ~/.docker
echo '{
  "experimental": "enabled"
}' > ~/.docker/config.json
sudo systemctl start docker
echo '{
  "storage-driver": "devicemapper",
  "storage-opts": [
    "dm.thinp_percent=95",
    "dm.thinp_metapercent=1",
    "dm.thinp_autoextend_threshold=80",
    "dm.thinp_autoextend_percent=20"
  ],
  "dns": [
        "141.223.1.2",
        "1.0.0.1"
    ]
}' | sudo tee /etc/docker/daemon.json
sudo usermod -a -G docker $USER
sudo chmod 666 /var/run/docker.sock
cat ubuntu.tar.gz.part* > ubuntu.tar.gz
docker load < ubuntu.tar.gz
mkdir /mnt/sdd1/jeseok
mkdir /mnt/sdd1/jeseok/mysql
mkdir /mnt/sdb1/jeseok
mkdir /mnt/sdb1/jeseok/lib
mkdir /mnt/sdc1/jeseok
mkdir /mnt/sdc1/jeseok/root
newgrp docker