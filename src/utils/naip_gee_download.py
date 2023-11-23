import os
import ee
from tqdm import tqdm
import geopandas as gpd


def export_image(polygon, index, max_dimension=2048):
    roi = ee.Geometry.Polygon(polygon, proj="EPSG:4326")

    collection = (ee.ImageCollection('USDA/NAIP/DOQQ')
                  .filterBounds(roi.bounds())
                  .filterDate('2018-01-01T00:00:00', '2023-12-31T23:59:59')
                  .sort('system:time_start', False))

    image = collection.select(["R", "G", "B"]).mosaic()

    task = ee.batch.Export.image.toCloudStorage(
        fileNamePrefix=f"Landmarks/NAIP_Scaled/NAIP_{index}",
        image=image,
        description=f"NAIP_{index}",
        bucket="vleo_benchmark",
        maxPixels=1e13,
        region=roi.bounds(),
        crs="EPSG:3857",
        dimensions=max_dimension,
    )
    task.start()

    print(f"Started download task for polygon {index}.")

    return task


def main():
    ee.Initialize(project=os.environ["EE_PROJECT"])

    shapefile_path = './data/Landmarks/Landmarks_OSM_all_unions_polygons_convexhulls_metadata_us.zip'
    shp = gpd.read_file(shapefile_path).to_crs(4326)
    shp_bounds = shp.to_crs(3857).bounds.to_numpy()

    tasks = []
    for idx, geometry, bbox in tqdm(zip(shp["entity_id"], shp["geometry"], shp_bounds), total=len(shp)):
        geom = geometry.__geo_interface__['coordinates'][0]
        tasks.append(export_image(geom, idx))

    print("All tasks initiated.")


if __name__ == "__main__":
    main()
