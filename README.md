
docker build --platform linux/amd64 --tag wngasinur7/runpod-sol-vanity:latest .


docker push wngasinur7/runpod-sol-vanity:latest

python main.py search-pubkey --starts-with SoL