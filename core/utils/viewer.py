import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from typing import Union
from matplotlib.patches import Patch



class Viewer:
    
    def __init__(self):
        pass

    def _get_legend(self, binary_mode: bool = False, error_mode: bool = False):
        if binary_mode:
            return [
                Patch(facecolor='white', edgecolor='black', label='Vessels'),
                Patch(facecolor='black', edgecolor='black', label='Background'),
            ]
        else:
            return [
                Patch(facecolor='white', edgecolor='black', label='True Positive'),
                Patch(facecolor='black', edgecolor='black', label='True Negative'),
                Patch(facecolor=(100/255, 100/255, 1.0), edgecolor='black', label='False Positive'),
                Patch(facecolor=(1.0, 100/255, 100/255), edgecolor='black', label='False Negative'),
            ]

    def _normalize_inputs(self, images, titles):
        if isinstance(images, np.ndarray):
            images = [images]
        num_images = len(images)
        if isinstance(titles, str):
            titles = [titles]
        if titles is None:
            titles = [f"Image {i+1}" for i in range(num_images)]
        elif len(titles) != num_images:
            raise ValueError("Number of titles must match number of images")
        return images, titles, num_images


    def is_binary_image(self, img: np.ndarray) -> bool:
        unique_vals = np.unique(img)
        return np.array_equal(unique_vals, [0, 1]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [1])


    def display_images(self, 
        images: Union[list[np.ndarray], np.ndarray],
        titles: Union[list[str], str] = None,
        cmap='gray', 
        error_mode: bool =False,
        binary_mode: bool = False,
    ):
        images, titles, num_images = self._normalize_inputs(images, titles)
        max_cols = 5
        ncols = max_cols
        nrows = math.ceil((num_images + (1 if error_mode else 0)) / max_cols)
        fig_width = 2.2 * ncols
        fig_height = 2.0 * nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
        axs = np.atleast_2d(axs)
        axs_flat = axs.flatten()

        for i, img in enumerate(images):
            ax = axs_flat[i]
            # used_cmap = None if ((not binary_mode or error_mode) else cmap
            used_cmap = None if error_mode else cmap
            im = ax.imshow(img, cmap=used_cmap)
            ax.set_title(titles[i], fontsize=9)
            ax.axis('off')
            if not error_mode or not binary_mode:
                cbar = fig.colorbar(im, ax=ax, shrink=0.9)
                cbar.ax.tick_params(labelsize=6)

        # Legend
        if error_mode or binary_mode:
            ax = axs_flat[num_images]
            ax.axis('off')
            legend_elements = self._get_legend(binary_mode, error_mode)
            ax.legend(handles=legend_elements, loc='center', fontsize=10)

        # Hide unused axes
        for j in range(num_images + (1 if error_mode else 0), len(axs_flat)):
            axs_flat[j].axis('off')

        plt.tight_layout()

        return fig




    
    def display_histograms(self, images: list[np.ndarray], titles: list[str] = None, bins: int = 50, density: bool = False, color='blue'):

        num_images = len(images)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.atleast_1d(axs).flatten()

        for i, image in enumerate(images):
            axs[i].hist(image.ravel(), bins=bins, density=density, color=color, alpha=0.7)
            axs[i].set_title(titles[i])
            axs[i].set_xlabel('Intensity')
            axs[i].set_ylabel('Density' if density else 'Frequency')
            axs[i].grid(True)

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()

    
        return fig
    

    

    def display_mip(self, images: Union[list[np.ndarray], np.ndarray],
                    titles: Union[list[str], str] = None, cmap='gray'):

        images, titles, num_images = self._normalize_inputs(images, titles)

        # Number of cols
        if num_images <= 5:
            cols = 1
        else:
            cols = 2

        rows = math.ceil(num_images / cols)
        total_columns = 4 * cols

        fig, axs = plt.subplots(nrows=rows, ncols=total_columns, figsize=(8 * cols, 1.5 * rows))

        if rows == 1:
            axs = axs[np.newaxis, :]
        if total_columns == 1:
            axs = axs[:, np.newaxis]

        axs = np.atleast_2d(axs)

        for idx, image in enumerate(images):
            if image.ndim != 3:
                raise ValueError(f"Image {idx} must be 3D. Current shape: {image.shape}.")

            col_idx = idx // rows
            row_idx = idx % rows
            base_col = col_idx * 4

            mips = [np.max(image, axis=ax) for ax in (0, 1, 2)]
            axis_labels = ['x-axis', 'y-axis', 'z-axis']

            axs[row_idx, base_col].text(0.5, 0.5, titles[idx], rotation=90, fontsize=9, fontweight='bold',
                                        verticalalignment='center', horizontalalignment='center')
            axs[row_idx, base_col].axis('off')

            for j, (mip, label) in enumerate(zip(mips, axis_labels)):
                ax = axs[row_idx, base_col + j + 1]
                im = ax.imshow(mip, cmap=cmap)
                ax.axis('off')
                cbar = fig.colorbar(im, ax=ax, shrink=0.9)
                cbar.ax.tick_params(labelsize=8)
                if row_idx == 0:
                    ax.set_title(f'{label}', fontsize=9)

        plt.subplots_adjust(top=0.98, bottom=0.02, right=0.95, left=0.01, wspace=0.2, hspace=0.1)


        return fig




    def display_volume(self, volume, threshold=0, cmap='plasma'):
        x, y, z = np.where(volume > threshold)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, s=0.5, marker='.', c=volume[x, y, z], cmap=cmap)

        ax.set(xlabel='X', ylabel='Y', zlabel='Z', title=f'3D plot with threshold {threshold}')
        fig.colorbar(sc, ax=ax, shrink=0.6, label='Intensity')
        plt.tight_layout()
        plt.show()
        return fig


    def _get_frame(self, img, d, idx):
        if img.ndim == 4:
            if d == 0:
                return img[idx, :, :, :]
            elif d == 1:
                return img[:, idx, :, :]
            else:
                return img[:, :, idx, :]
        else:
            return img[idx, :, :] if d == 0 else img[:, idx, :] if d == 1 else img[:, :, idx]
    
    
    def display_slices(self, 
        volumes: Union[list[np.ndarray], np.ndarray],
        titles: Union[list[str], str] = None,
        interval=10, 
        cmap='gray', 
        error_mode: bool = False,
        binary_mode: bool = False,
    ):
        
        volumes, titles, num_volumes = self._normalize_inputs(volumes, titles)
        directions = [0, 1, 2]

        if num_volumes <= 5:
            cols = 1
        else:
            cols = 2

        rows = math.ceil(num_volumes / cols)
        ncols_per_volume = 5 if error_mode or binary_mode else 4
        total_columns = ncols_per_volume * cols
        
        fig, axs = plt.subplots(nrows=rows, ncols=total_columns, figsize=(8 * cols, 1.8 * rows))
        axs = np.atleast_2d(axs)

        plots = []
        subplot_titles = []

        num_frames = min(min(volume.shape[d] for d in directions) for volume in volumes)

        for idx, volume in enumerate(volumes):
            
            if error_mode:
                if volume.ndim == 4 and volume.shape[-1] == 3:
                    is_rgb = True
                elif volume.ndim == 3:
                    is_rgb = False
                else:
                    raise ValueError(f"Volume {idx} must be 3D or 4D RGB. Got shape: {volume.shape}")
            else:
                if volume.ndim != 3:
                    raise ValueError(f"Volume {idx} must be 3D. Got shape: {volume.shape}")
                is_rgb = False
            vmin = np.min(volume)
            vmax = np.max(volume)

            col_idx = idx // rows
            row_idx = idx % rows
            base_col = col_idx * ncols_per_volume

            used_cmap = None if is_rgb else cmap

            ax_title = axs[row_idx, base_col]
            ax_title.text(0.5, 0.5, titles[idx], rotation=90, fontsize=10, fontweight='bold',
                          verticalalignment='center', horizontalalignment='center')
            ax_title.axis('off')

            volume_plots = []
            volume_titles = []

            for j, d in enumerate(directions):
                ax = axs[row_idx, base_col + j + 1]
                frame = self._get_frame(volume, d, 0)
                if is_rgb:
                    im = ax.imshow(frame)
                else:
                    im = ax.imshow(frame, cmap=used_cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')
                title = ax.set_title(f'Axis {d} - Slice 1/{volume.shape[d]}', fontsize=8)
                if not error_mode or binary_mode:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
                    cbar.ax.tick_params(labelsize=6)

                volume_plots.append(im)
                volume_titles.append(title)

            plots.append(volume_plots)
            subplot_titles.append(volume_titles)

            # Legend
            if error_mode or binary_mode:
                ax_legend = axs[row_idx, base_col + 4]
                ax_legend.axis('off')
                legend_elements = self._get_legend(binary_mode, error_mode)
                ax_legend.legend(handles=legend_elements, loc='center left', fontsize=10)

        # Update animation
        def update(frame_idx):
            artists = []
            for i, volume in enumerate(volumes):
                for d, im in enumerate(plots[i]):
                    if frame_idx < volume.shape[d]:
                        new_slice = self._get_frame(volume, d, frame_idx)
                        im.set_array(new_slice)
                        subplot_titles[i][d].set_text(f'Axis {d} - Slice {frame_idx + 1}/{volume.shape[d]}')
                        artists.append(im)
                        artists.append(subplot_titles[i][d])
            return artists

        plt.subplots_adjust(top=0.98, bottom=0.02, right=0.95, left=0.01, wspace=0.2, hspace=0.1)

        anim = ani.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

        return anim
    

    
        


