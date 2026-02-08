import numpy as np
db = np.load("default_database.npz", allow_pickle=True)

print(db.files)

star_table = db["star_table"]
print(type(star_table))
print(star_table.shape)
print(star_table.dtype)
print(star_table.dtype.names)
print(star_table[:5])

print("---")

pattern_catalog = db["pattern_catalog"]
print(type(pattern_catalog))
print(pattern_catalog.shape)
print(pattern_catalog.dtype)
print(pattern_catalog.dtype.names)
print(pattern_catalog[:10])

print("---")

star_catalog_IDs = db["star_catalog_IDs"]
print(type(star_catalog_IDs))
print(star_catalog_IDs.shape)
print(star_catalog_IDs.dtype)
print(star_catalog_IDs.dtype.names)
print(star_catalog_IDs[:10])

print("---")

props_packed = db["props_packed"]
for name in props_packed.dtype.names:
    print(name, "=", props_packed[name])