curl -fsSL https://www.scootersoftware.com/DEB-GPG-KEY-scootersoftware.asc | sudo tee /etc/apt/trusted.gpg.d/DEB-GPG-KEY-scootersoftware.asc &&
curl -fsSL https://www.scootersoftware.com/scootersoftware.list | sudo tee /etc/apt/sources.list.d/scootersoftware.list &&
sudo apt update &&
sudo apt -y install bcompare
