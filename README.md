# Depth Wizard

SIH — elevation extraction & 3D terrain viewer from satellite imagery

## Running with Docker (recommended)

Requires Docker Desktop installed and running.

docker build -t depth-wizard .
docker run -p 8000:8000 depth-wizard


Then open http://127.0.0.1:8000 in your browser. Upload a regular photo
for relative depth, or a georeferenced GeoTIFF for real elevation in meters.