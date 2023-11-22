import ee
from tqdm import tqdm
import geopandas as gpd

ee.Initialize(project="ee-chenhui5")

shapefile_path = '/home/danielz/PycharmProjects/vleo-bench/data/Landmarks/all_unions_polygons_convexhulls_metadata_us.gpkg'
shp = gpd.read_file(shapefile_path).to_crs(4326).iloc[[282, 283, 284]]
shp_bounds = shp.to_crs(3857).bounds.to_numpy()


def export_image(geom, idx, bounds, target_width=1920, target_height=1080):
    roi = ee.Geometry.Polygon(geom)
    xmin, ymin, xmax, ymax = bounds
    bbox_width = (xmax - xmin) / 0.6
    bbox_height = (ymax - ymin) / 0.6
    
    aspect_ratio = bbox_width / bbox_height
    if aspect_ratio > (target_width / target_height):
        # Bounding box is wider than the target aspect ratio
        scale = bbox_width / target_width
    else:
        # Bounding box is taller than the target aspect ratio
        scale = bbox_height / target_height
        
    scale = max(scale * 0.6, 0.6)    
    
    collection = (ee.ImageCollection('USDA/NAIP/DOQQ')
                   .filterBounds(roi)
                   # .filterDate('2019-01-01T00:00:00', '2023-12-31T23:59:59')
                   .sort('system:time_start', False))

    collection_info = collection.getInfo()
    if not collection_info["bands"]:
        print("Empty image!")
        print(roi.getInfo())
        return

    image = collection.select(["R", "G", "B"]).mosaic()
    
    task = ee.batch.Export.image.toCloudStorage(
        fileNamePrefix=f"Landmarks/NAIP_Scaled/NAIP_{idx}",
        image=image.clip(roi.bounds()),
        description=f"NAIP_{idx}",
        bucket="vleo_benchmark",
        maxPixels=1e13,
        region=roi, # .getInfo()
        scale=scale,
    )
    task.start()

    print(f"Started download task for polygon {idx}.")
    
    return task


tasks = []
for idx, geometry, bbox in tqdm(zip(shp["wikidata_entity_id"], shp["geometry"], shp_bounds), total=len(shp)):
    geom = geometry.__geo_interface__['coordinates']
    tasks.append(export_image(geom, idx, bbox))

print("All tasks initiated.")
