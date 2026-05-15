import pandas as pd
import json

def find_top_districts():
    df = pd.read_csv('ICRISAT-District Level Data.csv')
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year]
    
    crops = {
        "rice": "RICE PRODUCTION (1000 tons)",
        "wheat": "WHEAT PRODUCTION (1000 tons)",
        "sugarcane": "SUGARCANE PRODUCTION (1000 tons)",
        "cotton": "COTTON PRODUCTION (1000 tons)"
    }
    
    with open("district_centroids.json", "r") as f:
        centroids = json.load(f)
        
    # Flat list of all districts in centroids for fuzzy matching or simple lookup
    centroid_districts = {}
    for state, dists in centroids.items():
        for d_name in dists.keys():
            centroid_districts[d_name.upper()] = (state, d_name)

    top_districts = {}
    
    for crop, col in crops.items():
        # Get top 30 districts globally for this crop
        top = df_latest.sort_values(by=col, ascending=False).head(50)
        
        valid_districts = []
        for _, row in top.iterrows():
            d_name = str(row['Dist Name']).upper()
            s_name = str(row['State Name']).upper()
            
            # Try to match in centroids
            match = None
            if d_name in centroid_districts:
                match = centroid_districts[d_name]
            else:
                # Try with state name too
                # Some districts might have same name in different states
                pass # For now, simple name match
            
            if match:
                valid_districts.append({
                    "state": match[0],
                    "district": match[1],
                    "production": row[col]
                })
                
        top_districts[crop] = valid_districts[:20] # Keep top 20 valid ones
        
    with open("top_producing_districts.json", "w") as f:
        json.dump(top_districts, f, indent=4)
        
    for crop, dists in top_districts.items():
        print(f"{crop}: Found {len(dists)} valid districts")

if __name__ == "__main__":
    find_top_districts()
