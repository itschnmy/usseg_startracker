from dataclasses import dataclass
from math import asin, sqrt, sin, cos, floor, pi
import struct
from typing import List, Tuple, Optional
import numpy as np


K_VECTOR_MAGIC_NUMBER = 0x4253F009

@dataclass
class CatalogStar:
    raj2000: float
    dej2000: float
    magnitude: float
    weird: bool = False
    name: int = -1


@dataclass
class KVectorPair:
    index1: int
    index2: int
    distance: float

def filter_catalog_by_magnitude(catalog, max_mag=4.0):
    return [star for star in catalog if star.magnitude <= max_mag]

def load_tetra_catalog(npz_path: str) -> List[CatalogStar]:
    data = np.load(npz_path)

    star_table = data["star_table"]
    star_ids = data["star_catalog_IDs"]

    catalog = []
    for row, sid in zip(star_table, star_ids):
        ra = float(row[0])
        dec = float(row[1])
        mag = float(row[5])

        catalog.append(
            CatalogStar(
                raj2000=ra,
                dej2000=dec,
                magnitude=mag,
                weird=False,
                name=int(sid)
            )
        )

    return catalog


def great_circle_distance(ra1: float, de1: float, ra2: float, de2: float) -> float:
    """
    Python version of GreatCircleDistance() from attitude-utils.cpp
    Input/output units: radians
    """
    return 2.0 * asin(
        sqrt(
            sin(abs(de1 - de2) / 2.0) ** 2
            + cos(de1) * cos(de2) * sin(abs(ra1 - ra2) / 2.0) ** 2
        )
    )


def build_kvector_database(
    catalog: List[CatalogStar],
    min_distance: float,
    max_distance: float,
    num_bins: int
) -> bytes:
    """
    Python version of BuildKVectorDatabase(...)

    Returns:
        bytes object containing the packed k-vector database.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")
    if not (0.0 <= min_distance < max_distance <= pi):
        raise ValueError("Require 0 <= min_distance < max_distance <= pi")

    k_vector = [0] * (num_bins + 1)
    pairs: List[KVectorPair] = []

    bin_width = (max_distance - min_distance) / num_bins

    # Generate all valid star pairs
    for i in range(len(catalog)):
        for k in range(i + 1, len(catalog)):
            dist = great_circle_distance(
                catalog[i].raj2000, catalog[i].dej2000,
                catalog[k].raj2000, catalog[k].dej2000
            )

            if min_distance <= dist <= max_distance:
                pairs.append(KVectorPair(i, k, dist))

    # Sort by distance
    pairs.sort(key=lambda p: p.distance)

    # Build k-vector bins
    last_bin = 0
    for i, pair in enumerate(pairs):
        this_bin = int(floor((pair.distance - min_distance) / bin_width))
        if this_bin == num_bins:
            this_bin -= 1

        if not (0 <= this_bin < num_bins):
            raise ValueError(f"Computed invalid bin index: {this_bin}")

        for b in range(last_bin, this_bin + 1):
            k_vector[b + 1] = i
        last_bin = this_bin

    for b in range(last_bin + 1, num_bins):
        k_vector[b + 1] = k_vector[last_bin + 1]

    # Pack database as bytes
    # Layout matches the C++ comment:
    # magic(int32), numPairs(int32), minDistance(float), maxDistance(float), numBins(int32),
    # pairs as int16,int16,... then bins as int32...
    output = bytearray()

    output += struct.pack("<i", K_VECTOR_MAGIC_NUMBER)
    output += struct.pack("<i", len(pairs))
    output += struct.pack("<f", min_distance)
    output += struct.pack("<f", max_distance)
    output += struct.pack("<i", num_bins)

    for pair in pairs:
        output += struct.pack("<hh", pair.index1, pair.index2)

    for bin_val in k_vector:
        output += struct.pack("<i", bin_val)

    return bytes(output)


class KVectorDatabase:
    """
    Python version of the C++ KVectorDatabase parser.
    """

    def __init__(self, database_bytes: bytes):
        offset = 0

        self.magic, = struct.unpack_from("<i", database_bytes, offset)
        offset += 4
        if self.magic != K_VECTOR_MAGIC_NUMBER:
            raise ValueError("Invalid k-vector database magic number")

        self.num_pairs, = struct.unpack_from("<i", database_bytes, offset)
        offset += 4

        self.min_distance, = struct.unpack_from("<f", database_bytes, offset)
        offset += 4

        self.max_distance, = struct.unpack_from("<f", database_bytes, offset)
        offset += 4

        self.num_bins, = struct.unpack_from("<i", database_bytes, offset)
        offset += 4

        if self.min_distance <= 0.0:
            raise ValueError("min_distance must be > 0")
        if self.max_distance <= self.min_distance:
            raise ValueError("max_distance must be > min_distance")

        # Read pairs
        self.pairs: List[Tuple[int, int]] = []
        for _ in range(self.num_pairs):
            i1, i2 = struct.unpack_from("<hh", database_bytes, offset)
            offset += 4
            self.pairs.append((i1, i2))

        # Read bins
        self.bins: List[int] = []
        for _ in range(self.num_bins + 1):
            val, = struct.unpack_from("<i", database_bytes, offset)
            offset += 4
            self.bins.append(val)

    def bin_for_distance(self, distance: float) -> int:
        bin_width = (self.max_distance - self.min_distance) / self.num_bins
        result = int(floor((distance - self.min_distance) / bin_width))
        if result == self.num_bins:
            return self.num_bins - 1
        return result

    def find_possible_star_pairs_approx(
        self,
        min_query_distance: float,
        max_query_distance: float
    ) -> List[Tuple[int, int]]:
        """
        Python version of FindPossibleStarPairsApprox(...)

        Returns:
            list of (index1, index2) catalog pairs
        """

        if not (max_query_distance > min_query_distance):
            raise ValueError("Require max_query_distance > min_query_distance")

        if (
            min_query_distance < self.min_distance
            or min_query_distance > self.max_distance
            or max_query_distance < self.min_distance
            or max_query_distance > self.max_distance
        ):
            return []

        lower_bin = self.bin_for_distance(min_query_distance)
        upper_bin = self.bin_for_distance(max_query_distance)

        lower_pair = self.bins[lower_bin]
        upper_pair = self.bins[upper_bin + 1]

        return self.pairs[lower_pair:upper_pair]