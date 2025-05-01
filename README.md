docker build --platform linux/amd64 --tag wngasinur7/runpod-sol-vanity:beta .

docker push wngasinur7/runpod-sol-vanity:beta

python main.py search-pubkey --starts-with SoL
