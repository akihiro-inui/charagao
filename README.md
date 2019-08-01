## Installation

# A. Use docker-compose 
docker-compose up --build

# B. Use docker without docker-compose
docker build -t image_similarity:image_similarity .
docker run -p 5000:5000 image_similarity

## Image Collector (e.g. collect 100 image files of banana)
python src/datapreprocess/image_collector.py -k banana -o ./data -n 100

## Main Script (Image Similarity Evaluation)
python src/image_similarity_evaluation.py