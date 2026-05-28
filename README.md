docker build --platform linux/amd64 --tag wngasinur7/runpod-sol-vanity:1.6 .

docker push wngasinur7/runpod-sol-vanity:1.6

python main.py search-pubkey --starts-with SoL
