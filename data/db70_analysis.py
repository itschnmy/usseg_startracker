import numpy as np

data = np.load("db_70-1deg.npz", allow_pickle=True)

for key in data.files:
    print(key, data[key].shape)

print("Keys inside database:")
print(data.files)

stars = data["star_table"]
print("Number of stars in catalog:", len(stars))

print("Keys:", data.files)
print("Saved star_table size:", data["star_table"].shape)

props = data["props_packed"]
print("star_catalog:", props["star_catalog"][()])
print("max_fov:", props["max_fov"][()])
print("min_fov:", props["min_fov"][()])
print("pattern_stars_per_fov:", props["pattern_stars_per_fov"][()])
print("verification_stars_per_fov:", props["verification_stars_per_fov"][()])
print("star_max_magnitude:", props["star_max_magnitude"][()])
print("range_ra:", props["range_ra"][()])
print("range_dec:", props["range_dec"][()])