import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm


anchor_markers = ['^', 's', 'D', 'v', 'P', '*', 'X', 'o', '<', '>']
anchor_offsets = [(-4, -4), (4, -4), (-4, 4), (4, 4), (0, -6),
                  (0, 6), (-6, 0), (6, 0), (-3, 3), (3, -3)]


def visualize_assignments(assigned_labels, bboxes, anchor_boxes, images=None, batch_idx=0, img_size=(416, 416),
                            image_stride=(8, 16, 32), figsizes=(16, 12, 8), save_dir='assigned_results', colormap='tab20',
                            markersizes=(4, 4, 8), fontsizes=(6, 6, 6), offset_scale=(2, 1, 1)):
    assert len(assigned_labels) == len(image_stride)
    cmap = plt.get_cmap(colormap)
    color_list = [cmap(i % cmap.N) for i in range(len(bboxes))]

    for k in range(len(assigned_labels)):
        assigned_tensor = assigned_labels[k]
        B, A, H, W, D = assigned_tensor.shape
        grid_h = img_size[0] // image_stride[k]
        grid_w = img_size[1] // image_stride[k]

        # Adaptive figure size
        if img_size[0] >= img_size[1]:
            figh = figsizes[k]
            figw = int(img_size[0] / img_size[1] * figsizes[k])
        else:
            figw = figsizes[k]
            figh = int(img_size[1] / img_size[0] * figsizes[k])
        figsize = (figw, figh)

        num_images = B if images is None else len(images)
        for b_idx in tqdm(range(num_images), desc=f"Layer {k}"):
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            ax.set_xlim(0, img_size[1])
            ax.set_ylim(img_size[0], 0)
            ax.set_title(f"Image {b_idx}: Anchor Assignments")
            ax.set_xticks(np.arange(0, img_size[1] + 1, image_stride[k]))
            ax.set_yticks(np.arange(0, img_size[0] + 1, image_stride[k]))
            ax.set_xticklabels([f'{i}\n({i / grid_w:.2f})' if k > 1 else f'{i}' for i in range(grid_w + 1)])
            ax.set_yticklabels([f'{i}\n({i / grid_h:.2f})' if k > 1 else f'{i}' for i in range(grid_h + 1)])
            ax.grid(color='gray', linestyle='--', linewidth=0.5)

            obj_bboxes = bboxes[bboxes[:, 0] == b_idx]
            obj_colors = [color_list[i] for i in range(len(obj_bboxes))]

            for i, gt in enumerate(obj_bboxes):
                _, cx, cy, w, h = gt.cpu().numpy()
                px = cx * img_size[0]
                py = cy * img_size[1]
                ax.plot(px, py, marker='o', markersize=markersizes[k], color=obj_colors[i])
                text = f"{w:.2f}, {h:.2f})"
                ax.text(px + 3, py + 3, '--(', fontsize=fontsizes[k], color=obj_colors[i], ha='left', va='bottom')
                ax.text(px + 6, py + 3, text, fontsize=fontsizes[k], color='black', ha='left', va='bottom')

            labels = assigned_tensor[b_idx]
            conf_mask = labels[..., 0] > 0
            a_idx, i_idx, j_idx = torch.nonzero(conf_mask, as_tuple=True)
            if a_idx.numel() == 0:
                plt.close()
                continue

            assigned_data = labels[a_idx, i_idx, j_idx]
            assigned_centers = assigned_data[:, 1:3]

            same_image_gt = obj_bboxes[:, 1:3]
            if same_image_gt.numel() > 0:
                distances = torch.cdist(assigned_centers, same_image_gt)
                matched_obj = torch.argmin(distances, dim=1)
                colors = np.array([obj_colors[i] for i in matched_obj.tolist()])
            else:
                colors = np.tile([[0.5, 0.5, 0.5, 1.0]], (assigned_data.size(0), 1))

            cx_img = (j_idx + 0.5) * image_stride[k]
            cy_img = (i_idx + 0.5) * image_stride[k]
            offsets = np.array(anchor_offsets)[np.array(a_idx) % len(anchor_offsets)]
            dx, dy = offsets[:, 0] // offset_scale[k], offsets[:, 1] // offset_scale[k]
            cx_img = cx_img + dx
            cy_img = cy_img + dy

            # Plot per anchor marker
            for anchor_id in range(A):
                marker_mask = (a_idx == anchor_id).cpu().numpy()
                if marker_mask.any():
                    ax.scatter(
                        cx_img[marker_mask], cy_img[marker_mask],
                        color=colors[marker_mask],
                        s=markersizes[k] ** 2,
                        marker=anchor_markers[anchor_id % len(anchor_markers)],
                        zorder=10,
                        edgecolors='none'
                    )

            # Set legend
            legend_elements = [Line2D([0], [0], marker='o', color='gray', label='GT',
                                      markerfacecolor='gray', markersize=8, linestyle='None')]
            for i in range(min(A, len(anchor_markers))):
                anchor_label = f"({anchor_boxes[k][i][0] / img_size[0]:.2f}, {anchor_boxes[k][i][1] / img_size[1]:.2f})"
                legend_elements.append(Line2D([0], [0], marker=anchor_markers[i], color='black',
                                              label=anchor_label, markerfacecolor='black', markersize=8, linestyle='None'))
            legend = ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)
            legend.set_zorder(100)
            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"batch{batch_idx}_img{b_idx}_layer{k}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=300)

            plt.close()