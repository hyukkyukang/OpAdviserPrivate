echo "# Generated by NetworkManager
search cluster. cluster
nameserver 141.223.1.2
#nameserver 1.0.0.1" | sudo tee /etc/resolv.conf
mkdir ~/.docker
echo '{
  "experimental": "enabled"
}' > ~/.docker/config.json
sudo usermod -a -G docker $USER
newgrp docker
sudo systemctl start docker
docker load < ubuntu.tar.gz
git fetch
git checkout -b ground_truth