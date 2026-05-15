import shapefile
from shapely.geometry import shape
import json
from pyproj import Transformer, CRS

def clean_name(name):
    if not isinstance(name, str):
        return name
    return name.replace('>', 'A').strip()

def extract_centroids(shp_path, output_json):
    sf = shapefile.Reader(shp_path)
    data = {}
    
    # Read projection
    prj_path = shp_path.replace('.shp', '.prj')
    with open(prj_path, 'r') as f:
        prj_text = f.read()
    
    source_crs = CRS.from_string(prj_text)
    target_crs = CRS.from_epsg(4326) # WGS84
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # Get index of District and STATE fields
    fields = [f[0] for f in sf.fields[1:]] # Skip DeletionFlag
    dist_idx = fields.index("District")
    state_idx = fields.index("STATE")
    
    for sr in sf.shapeRecords():
        dist_name = clean_name(sr.record[dist_idx])
        state_name = clean_name(sr.record[state_idx])
        
        geom = shape(sr.shape.__geo_interface__)
        centroid = geom.centroid
        
        # Transform coordinates
        lon, lat = transformer.transform(centroid.x, centroid.y)
        
        if state_name not in data:
            data[state_name] = {}
        
        data[state_name][dist_name] = {
            "lat": float(lat),
            "lon": float(lon)
        }
        
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Extracted {len(sf)} districts and saved to {output_json}")

if __name__ == "__main__":
    extract_centroids("DISTRICT_BOUNDARY.shp", "district_centroids.json")
