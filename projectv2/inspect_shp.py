import shapefile

def inspect_shapefile(path):
    sf = shapefile.Reader(path)
    print(f"Fields: {sf.fields}")
    # Print first few records to see data
    for i, record in enumerate(sf.records()[:5]):
        print(f"Record {i}: {record}")

if __name__ == "__main__":
    inspect_shapefile("DISTRICT_BOUNDARY.shp")
