import ee
from tqdm import tqdm
import geopandas as gpd

ee.Initialize(project="ee-chenhui5")

shapefile_path = './data/Landmarks/Landmarks_OSM_all_unions_polygons_convexhulls_metadata_us.zip'
shp = gpd.read_file(shapefile_path).to_crs(4326).iloc[[282, 283, 284]]
shp_bounds = shp.to_crs(3857).bounds.to_numpy()


def export_image(polygon, index, max_dimension=1024):
    roi = ee.Geometry.Polygon(polygon)

    collection = (ee.ImageCollection('USDA/NAIP/DOQQ')
                  .filterBounds(roi)
                  .filterDate('2019-01-01T00:00:00', '2023-12-31T23:59:59')
                  .sort('system:time_start', False))

    collection_info = collection.getInfo()
    if not collection_info["bands"]:
        print("Empty image!")
        print(roi.getInfo())
        return

    image = collection.select(["R", "G", "B"]).mosaic().clipToBoundsAndScale(roi, maxDimension=max_dimension)
    image_scale = image.select('R').projection().nominalScale()

    task = ee.batch.Export.image.toCloudStorage(
        fileNamePrefix=f"Landmarks/NAIP_Scaled/NAIP_{index}",
        image=image,
        description=f"NAIP_{index}",
        bucket="vleo_benchmark",
        maxPixels=1e13,
        region=roi,
        scale=image_scale,
    )
    task.start()

    print(f"Started download task for polygon {index}.")

    return task


def main():
    tasks = []
    for idx, geometry, bbox in tqdm(zip(shp["entity_id"], shp["geometry"], shp_bounds), total=len(shp)):
        geom = geometry.__geo_interface__['coordinates']
        tasks.append(export_image(geom, idx, bbox))

    print("All tasks initiated.")


if __name__ == "__main__":
    main()
