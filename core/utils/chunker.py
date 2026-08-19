import numpy as np
from typing import Tuple, Literal, Optional

class Chunker:
    def __init__(
        self,
        mode: Literal["size", "count", "ratio"] = "size",
        chunk_size: Optional[Tuple[int, int, int]] = None,
        chunk_count: Optional[Tuple[int, int, int]] = None,
        overlap: Tuple[int, int, int] = (0, 0, 0),
    ):
        self.mode = mode
        self.chunk_count = chunk_count
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.padding = None


    def compute_chunk_size(self, volume_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        x, y, z = volume_shape

        if self.mode == "size":
            if self.chunk_size is None:
                self.chunk_size = volume_shape
            return self.chunk_size

        elif self.mode == "count":
            if self.chunk_count is None:
                self.chunk_count = [1, 1, 1]
            nx, ny, nz = self.chunk_count
            self.chunk_size = (x // nx, y // ny, z // nz)
            return self.chunk_size

        else:
            raise ValueError("Invalid mode. Choose from 'size', 'count' or 'ratio'.")


    def pad_volume(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim != 3:
            raise ValueError(f"Volume must be of shape 3. Current shape: {volume.shape}")
        cx, cy, cz = self.chunk_size
        x, y, z = volume.shape

        pad_x = (cx - x % cx) % cx
        pad_y = (cy - y % cy) % cy
        pad_z = (cz - z % cz) % cz

        self.padding = ((0, pad_x), (0, pad_y), (0, pad_z))
        return np.pad(volume, self.padding, mode='reflect')


    def chunk_volume(self, volume: np.ndarray) -> list:
        if volume.ndim != 3:
            raise ValueError(f"Volume must be of shape 3. Current shape: {volume.shape}")
        
        self.chunk_size = self.compute_chunk_size(volume.shape)
        
        padded_volume = self.pad_volume(volume)
        self.padded_volume_shape = padded_volume.shape
        
        x, y, z = padded_volume.shape
        cx, cy, cz = self.chunk_size
        ox, oy, oz = self.overlap

        chunks = []

        for i in range(0, x - ox, cx - ox):
            for j in range(0, y - oy, cy - oy):
                for k in range(0, z - oz, cz - oz):
                    chunk = padded_volume[
                        i:i + cx,
                        j:j + cy,
                        k:k + cz
                    ]
                    # Ignore incomplete chunks at the boundary
                    if chunk.shape == (cx, cy, cz):
                        chunks.append(((i, j, k), chunk))

        return chunks


    def unpad_volume(self, volume: np.ndarray) -> np.ndarray:
        if self.padding is None:
            return volume
        (px0, px1), (py0, py1), (pz0, pz1) = self.padding
        x_end = -px1 if px1 > 0 else None
        y_end = -py1 if py1 > 0 else None
        z_end = -pz1 if pz1 > 0 else None
        return volume[px0:x_end, py0:y_end, pz0:z_end]


    def unchunk_volume(self, chunks: list, volume_shape: Tuple[int, int, int]) -> np.ndarray:
        assert self.chunk_size is not None, "chunk_size must be defined"

        cx, cy, cz = self.chunk_size
        padded_volume = np.zeros(tuple(s + p[1] for s, p in zip(volume_shape, self.padding)), dtype=np.float32)
        count_mask = np.zeros_like(padded_volume, dtype=np.uint8)

        for (i, j, k), chunk in chunks:
            padded_volume[i:i + cx, j:j + cy, k:k + cz] += chunk
            count_mask[i:i + cx, j:j + cy, k:k + cz] += 1

        # Avoid 0 div
        count_mask = np.maximum(count_mask, 1)
        averaged_volume = padded_volume / count_mask

        return self.unpad_volume(averaged_volume)
