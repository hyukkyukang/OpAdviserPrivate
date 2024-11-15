echo "# Generated by NetworkManager
search cluster. cluster
nameserver 141.223.1.2
#nameserver 1.0.0.1" | sudo tee /etc/resolv.conf
mkdir -p ~/.docker
echo '{
  "experimental": "enabled"
}' > ~/.docker/config.json
sudo usermod -a -G docker $USER
newgrp docker
sudo systemctl start docker
docker load < ubuntu.tar.gz
echo '{
  "storage-driver": "devicemapper",
  "storage-opts": [
    "dm.min_free_space=5%"
  }
}' | sudo tee /etc/docker/daemon.json
