@echo off
echo Building Docker image...
docker build -t asia-southeast2-docker.pkg.dev/apnr-development-4ea10/apnr-ml/apnr-ml:v1 .

echo Pushing Docker image to Google Container Registry...
docker push asia-southeast2-docker.pkg.dev/apnr-development-4ea10/apnr-ml/apnr-ml:v1

echo Deploying to Cloud Run with unauthenticated access...
gcloud run deploy apnr-ml --image asia-southeast2-docker.pkg.dev/apnr-development-4ea10/apnr-ml/apnr-ml:v1 --allow-unauthenticated

echo Deployment complete!
pause
