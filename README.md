docker build --platform linux/amd64 --tag wngasinur7/runpod-sol-vanity:1.7 .

docker push wngasinur7/runpod-sol-vanity:1.7

python main.py search-pubkey --starts-with SoL
